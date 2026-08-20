import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from data.DLL_dataset import DLLDataset
from model.ddpm_modules.diffusion import GaussianDiffusion, make_beta_schedule, make_resampled_beta_schedule
from train import ExponentialMovingAverage, extract_network_state, match_prediction_mean_to_gt, optional_float, pad_to_multiple, stage_transitions


class DLLDatasetTests(unittest.TestCase):
    def test_pairs_and_deterministic_epoch_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = np.zeros((12, 12, 3), dtype=np.uint8)
            gt = np.full((12, 12, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(root / "sample_LOW.jpg"), low)
            cv2.imwrite(str(root / "sample_GT.jpg"), gt)
            dataset = DLLDataset(
                {
                    "root": str(root),
                    "expected_pairs": 1,
                    "patch_size": 8,
                    "use_crop": True,
                    "use_flip": True,
                    "use_rot": True,
                    "seed": 7,
                },
                train=True,
            )
            first = dataset[0]
            second = dataset[0]
            self.assertTrue(torch.equal(first["LQ"], second["LQ"]))
            self.assertEqual(tuple(first["LQ"].shape), (3, 8, 8))
            self.assertEqual(first["LQ"].min().item(), -1.0)
            self.assertEqual(first["GT"].max().item(), 1.0)


class ScheduleTests(unittest.TestCase):
    def test_resample_preserves_endpoint(self):
        options = {
            "schedule": "linear",
            "source_n_timestep": 500,
            "linear_start": 1e-4,
            "linear_end": 2e-2,
        }
        source = make_beta_schedule("linear", 500, 1e-4, 2e-2)
        target = make_resampled_beta_schedule(options, 512)
        self.assertEqual(len(target), 512)
        self.assertAlmostEqual(
            float(np.prod(1.0 - source)),
            float(np.prod(1.0 - target)),
            places=12,
        )

    def test_config_stages_halve_to_two(self):
        config_path = Path(__file__).parents[1] / "config" / "dll_train.json"
        config = json.loads(config_path.read_text())
        ladder = config["progressive"]["teacher_ladder"]
        self.assertEqual(ladder, [2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertEqual(config["progressive"]["iterations_per_stage"], 20000)

    def test_two_to_one_loss_handles_zero_endpoint(self):
        class Denoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(6, 3, 1)

            def forward(self, image, noise_level):
                del noise_level
                output = self.conv(image)
                return output, torch.zeros_like(output)

        schedule = {
            "schedule": "linear",
            "source_n_timestep": 4,
            "master_n_timestep": 4,
            "linear_start": 1e-4,
            "linear_end": 2e-2,
        }
        teacher = GaussianDiffusion(
            Denoiser(),
            image_size=8,
            num_timesteps=4,
            time_scale=1,
            schedule_opt=schedule,
        )
        student = Denoiser()
        batch = {"LQ": torch.rand(2, 3, 8, 8) * 2 - 1, "GT": torch.rand(2, 3, 8, 8) * 2 - 1}
        for _ in range(8):
            loss, parts = teacher.loss(batch, student, student_steps=2)
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("distill_loss", parts)
            loss.backward()
            student.zero_grad(set_to_none=True)


class CheckpointTests(unittest.TestCase):
    def test_restores_missing_legacy_resblock_mlp_alias(self):
        weight = torch.ones(4, 3)
        payload = {
            "state_dict": {
                "model.denoise_fn.downs.1.res_block.noise_func.noise_func.0.weight": weight,
            }
        }
        state = extract_network_state(payload)
        self.assertTrue(torch.equal(state["downs.1.res_block.mlp.1.weight"], weight))

    def test_extracts_ema_and_normalizes_network_prefix(self):
        payload = {
            "ema": {
                "betas": torch.ones(5),
                "denoise_fn.layer.weight": torch.ones(2, 2),
                "denoise_fn.layer.bias": torch.zeros(2),
            },
            "model": {"denoise_fn.layer.weight": torch.zeros(2, 2)},
        }
        state = extract_network_state(payload, prefer_ema=True)
        self.assertTrue(torch.equal(state["layer.weight"], torch.ones(2, 2)))
        self.assertIn("betas", state)

    def test_ema_round_trip(self):
        model = nn.Linear(2, 2)
        ema = ExponentialMovingAverage(model, 0.5)
        initial = ema.state_dict_cpu()
        with torch.no_grad():
            model.weight.add_(2)
        ema.update(model)
        saved = ema.state_dict_cpu()
        restored = ExponentialMovingAverage(model, 0.5)
        restored.load_state_dict(saved)
        self.assertEqual(set(restored.state_dict_cpu()), set(initial))
        self.assertTrue(torch.equal(restored.state_dict_cpu()["weight"], saved["weight"]))


class ProgressiveTrainingTests(unittest.TestCase):
    def test_transition_plan(self):
        self.assertEqual(stage_transitions(16), [(16, 8), (8, 4), (4, 2)])
        self.assertEqual(stage_transitions(2), [(4, 2)])

    def test_null_optional_metric_uses_default(self):
        self.assertEqual(optional_float(None, "-inf"), float("-inf"))

    def test_padding_preserves_original_region(self):
        image = torch.arange(3 * 17 * 19, dtype=torch.float32).reshape(1, 3, 17, 19)
        padded, original = pad_to_multiple(image, 16)
        self.assertEqual(original, (17, 19))
        self.assertEqual(tuple(padded.shape[-2:]), (32, 32))
        self.assertTrue(torch.equal(padded[..., :17, :19], image))

    def test_mean_matching_scales_each_image(self):
        prediction = torch.full((2, 3, 8, 8), -0.5)
        target = torch.stack(
            [torch.full((3, 8, 8), 0.0), torch.full((3, 8, 8), 0.5)]
        )
        matched = match_prediction_mean_to_gt(prediction, target)
        matched_mean = ((matched + 1.0) * 0.5).mean(dim=(1, 2, 3))
        target_mean = ((target + 1.0) * 0.5).mean(dim=(1, 2, 3))
        self.assertTrue(torch.allclose(matched_mean, target_mean))


if __name__ == "__main__":
    unittest.main()
