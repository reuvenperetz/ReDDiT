"""DLL progressive ReDDiT training with DDP, W&B, and exact resume."""

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from skimage.metrics import structural_similarity
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.DLL_dataset import DLLDataset
from model.ddpm_modules.diffusion import GaussianDiffusion
from model.ddpm_modules.unet import UNet
from utils.niqe import niqe


LOGGER = logging.getLogger("reddit")
NETWORK_PREFIXES = ("module.", "model.", "netG.", "generator.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dll_train.json")
    parser.add_argument("--dataset", default="config/dll.yml")
    parser.add_argument("--mode", choices=("train", "validate-teacher"), default="train")
    parser.add_argument("--output-dir")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--train-root")
    parser.add_argument("--val-root")
    parser.add_argument("--job-id", default="unknown")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-id")
    parser.add_argument("--resume", default="auto", help="Checkpoint path, auto, or none")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--validation-max-edge", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--stop-after-stage", action="store_true", help="Testing only: do not enter 2-NFE extensions")
    parser.add_argument("--learning-rate", type=float, help="Override the initial learning rate for a stage restart")
    parser.add_argument("--restart-reason", help="Reason recorded for an intentional stage restart")
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


def configure_logging(rank, output_dir):
    handlers = [logging.StreamHandler()]
    if rank == 0:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "train.log", mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_configs(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    with open(args.dataset, "r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)
    if dataset_config.get("dataset") != "DLL":
        raise ValueError("This training entry point intentionally supports DLL only")
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.teacher_checkpoint:
        config["teacher_checkpoint"] = args.teacher_checkpoint
    if args.train_root:
        dataset_config["datasets"]["train"]["root"] = args.train_root
    if args.val_root:
        dataset_config["datasets"]["val"]["root"] = args.val_root
    if args.wandb_name:
        config["wandb"]["name"] = args.wandb_name
    if args.wandb_id:
        config["wandb"]["id"] = args.wandb_id
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = float(args.learning_rate)
    if not config.get("teacher_checkpoint"):
        raise ValueError("--teacher-checkpoint is required")
    if not config.get("output_dir"):
        raise ValueError("--output-dir is required")
    if not dataset_config["datasets"]["train"].get("root"):
        raise ValueError("--train-root is required")
    if not dataset_config["datasets"]["val"].get("root"):
        raise ValueError("--val-root is required")
    if not config["wandb"].get("dir"):
        config["wandb"]["dir"] = str(Path(config["output_dir"]) / "wandb")
    config["job_id"] = args.job_id
    return config, dataset_config


def build_unet(config):
    opt = config["model"]["unet"]
    return UNet(
        in_channel=int(opt["in_channel"]),
        out_channel=int(opt["out_channel"]),
        inner_channel=int(opt["inner_channel"]),
        norm_groups=int(opt.get("norm_groups", 32)),
        channel_mults=tuple(opt["channel_multiplier"]),
        attn_res=tuple(opt["attn_res"]),
        res_blocks=int(opt["res_blocks"]),
        dropout=float(opt["dropout"]),
        image_size=int(opt["image_size"]),
    )


def _strip_network_prefix(key):
    changed = True
    while changed:
        changed = False
        for prefix in NETWORK_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    if key.startswith("denoise_fn."):
        key = key[len("denoise_fn.") :]
    return key


def extract_network_state(payload, prefer_ema=True):
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    candidates = ("ema", "model", "state_dict") if prefer_ema else ("model", "ema", "state_dict")
    state = None
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, dict):
            state = value
            break
    if state is None:
        state = payload
    normalized = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        normalized[_strip_network_prefix(key)] = value
    for key, value in list(normalized.items()):
        marker = ".res_block.noise_func.noise_func.0."
        if marker in key:
            legacy_key = key.replace(marker, ".res_block.mlp.1.")
            normalized.setdefault(legacy_key, value)
    return normalized


def validate_and_load_network(model, state, label):
    expected = model.state_dict()
    filtered = {key: value for key, value in state.items() if key in expected}
    missing = sorted(set(expected) - set(filtered))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key for key in filtered if tuple(filtered[key].shape) != tuple(expected[key].shape)
    )
    if missing or mismatched:
        raise RuntimeError(
            f"{label} is incompatible: missing={missing[:5]}, "
            f"shape_mismatch={mismatched[:5]}, ignored={unexpected[:5]}"
        )
    model.load_state_dict(filtered, strict=True)
    LOGGER.info("Loaded %s: %d tensors (%d non-network entries ignored)", label, len(filtered), len(unexpected))


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_cpu(model):
    if isinstance(model, DDP):
        model = model.module
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = state_dict_cpu(model)

    @torch.no_grad()
    def update(self, model):
        if isinstance(model, DDP):
            model = model.module
        for key, value in model.state_dict().items():
            source = value.detach()
            target = self.shadow[key]
            if target.device != source.device:
                target = target.to(source.device)
            if source.is_floating_point():
                target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                target.copy_(source)
            self.shadow[key] = target

    def load_state_dict(self, state):
        if set(state) != set(self.shadow):
            raise RuntimeError("EMA state keys do not match the student network")
        self.shadow = {key: value.detach().clone() for key, value in state.items()}

    def state_dict_cpu(self):
        return {key: value.detach().cpu() for key, value in self.shadow.items()}

    @contextlib.contextmanager
    def apply(self, model):
        if isinstance(model, DDP):
            model = model.module
        backup = state_dict_cpu(model)
        model.load_state_dict({key: value.to(next(model.parameters()).device) for key, value in self.shadow.items()})
        try:
            yield model
        finally:
            model.load_state_dict(backup)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_directory(output_dir, teacher_steps, student_steps, extension_cycle=0):
    if extension_cycle:
        name = f"extension_{extension_cycle:04d}_{teacher_steps}_to_{student_steps}"
    else:
        name = f"phase_{teacher_steps}_to_{student_steps}"
    return Path(output_dir) / "checkpoints" / name


def atomic_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def save_training_checkpoint(
    path,
    config,
    stage_index,
    teacher_steps,
    student_steps,
    stage_step,
    global_step,
    epoch,
    batch_in_epoch,
    student,
    ema,
    optimizer,
    scheduler,
    metrics,
    stage_complete,
    rank,
    world_size,
    teacher=None,
    metadata=None,
):
    rng = capture_rng_state()
    if world_size > 1:
        rng_by_rank = [None] * world_size
        dist.all_gather_object(rng_by_rank, rng)
    else:
        rng_by_rank = [rng]

    if rank == 0:
        payload = {
            "format_version": 2,
            "kind": "reddit_progressive_distillation",
            "source_commit": git_commit(),
            "stage_index": int(stage_index),
            "teacher_steps": int(teacher_steps),
            "student_steps": int(student_steps),
            "stage_step": int(stage_step),
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "stage_complete": bool(stage_complete),
            "student": state_dict_cpu(student),
            "ema": ema.state_dict_cpu(),
            "teacher": state_dict_cpu(teacher) if teacher is not None else {},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_by_rank": rng_by_rank,
            "metrics": metrics or {},
            "config": config,
            "metadata": metadata or {},
        }
        atomic_torch_save(payload, path)
        atomic_json_save({"checkpoint": str(Path(path).resolve())}, Path(config["output_dir"]) / "latest.json")
        LOGGER.info("Saved checkpoint: %s", path)
    barrier(world_size)
    return str(Path(path).resolve())


def update_manifest(config, **updates):
    path = Path(config["output_dir"]) / "manifest.json"
    manifest = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    manifest.update(
        {
            "commit": git_commit(),
            "job_id": config.get("job_id", "unknown"),
            "teacher_source": str(Path(config["teacher_checkpoint"]).resolve()),
            "updated_unix": time.time(),
        }
    )
    manifest.update(updates)
    atomic_json_save(manifest, path)


def find_resume_checkpoint(config, resume):
    if resume.lower() in ("none", "false", "off"):
        return None
    if resume != "auto":
        return Path(resume)
    pointer = Path(config["output_dir"]) / "latest.json"
    if not pointer.is_file():
        return None
    with open(pointer, "r", encoding="utf-8") as handle:
        return Path(json.load(handle)["checkpoint"])


def make_diffusion(config, network, num_steps, master_steps):
    loss = config["model"]["loss"]
    schedule = copy.deepcopy(config["model"]["schedule"])
    schedule["master_n_timestep"] = int(master_steps)
    if master_steps % num_steps:
        raise ValueError(f"{num_steps} does not divide master schedule {master_steps}")
    return GaussianDiffusion(
        network,
        image_size=config["model"]["unet"]["image_size"],
        num_timesteps=int(num_steps),
        time_scale=int(master_steps // num_steps),
        channels=3,
        w_snr=loss["w_refinement"],
        w_gt=loss["w_pixel"],
        w_lpips=loss["w_perceptual"],
        w_str=loss["w_structural"],
        conditional=True,
        schedule_opt=schedule,
    )


def pad_to_multiple(tensor, multiple):
    height, width = tensor.shape[-2:]
    pad_h = (-height) % int(multiple)
    pad_w = (-width) % int(multiple)
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (height, width)


def resize_for_smoke(low, gt, max_edge):
    if not max_edge or max(low.shape[-2:]) <= max_edge:
        return low, gt
    scale = float(max_edge) / max(low.shape[-2:])
    size = tuple(max(16, int(round(value * scale))) for value in low.shape[-2:])
    low = torch.nn.functional.interpolate(low, size=size, mode="bilinear", align_corners=False, antialias=True)
    gt = torch.nn.functional.interpolate(gt, size=size, mode="bilinear", align_corners=False, antialias=True)
    return low, gt


def match_prediction_mean_to_gt(output, ground_truth):
    prediction = ((output.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    target = ((ground_truth.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    prediction_mean = prediction.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    target_mean = target.mean(dim=(1, 2, 3), keepdim=True)
    return (prediction * (target_mean / prediction_mean)).clamp(0.0, 1.0).mul(2.0).sub(1.0)


def metric_values(output, ground_truth):
    output_rgb = np.clip((output.detach().float().cpu().squeeze(0).numpy().transpose(1, 2, 0) + 1) * 0.5, 0, 1)
    target_rgb = np.clip((ground_truth.detach().float().cpu().squeeze(0).numpy().transpose(1, 2, 0) + 1) * 0.5, 0, 1)
    mse = float(np.mean((output_rgb - target_rgb) ** 2))
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    ssim = float(structural_similarity(target_rgb, output_rgb, data_range=1.0, channel_axis=2))
    return psnr, ssim, output_rgb


@torch.no_grad()
def sample_mixed_precision(diffusion, low, initial_noise):
    model_dtype = next(diffusion.denoise_fn.parameters()).dtype
    x_t = initial_noise.float()
    for step in range(diffusion.num_timesteps, 0, -1):
        t_index = step * diffusion.time_scale
        s_index = (step - 1) * diffusion.time_scale
        alpha_t = diffusion.sqrt_alphas_cumprod[t_index].expand(low.shape[0], 1)
        eps, _ = diffusion.denoise_fn(
            torch.cat([low.to(model_dtype), x_t.to(model_dtype)], dim=1),
            alpha_t.to(model_dtype),
        )
        eps = eps.float()
        alpha_4d = diffusion._expand(alpha_t[:, 0])
        x_0 = diffusion.predict_start(x_t, alpha_4d, eps).clamp(-1.0, 1.0)
        eps = diffusion.predict_eps(x_t, x_0, alpha_4d)
        x_t = diffusion.sqrt_alphas_cumprod[s_index] * x_0 + diffusion.sqrt_one_minus_alphas_cumprod[s_index] * eps
    return x_t.clamp(-1.0, 1.0)


@torch.no_grad()
def evaluate_state(
    config,
    state,
    val_dataset,
    num_steps,
    device,
    rank,
    world_size,
    limit=0,
    max_edge=0,
    lpips_alex=None,
):
    model = build_unet(config)
    validate_and_load_network(model, state, f"{num_steps}-NFE validation model")
    precision = str(config["validation"].get("precision", "bfloat16"))
    model_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    model.to(device=device, dtype=model_dtype).eval()
    master_steps = int(config["model"]["schedule"]["master_n_timestep"])
    diffusion = make_diffusion(config, model, int(num_steps), master_steps).to(device).eval()
    count_total = min(len(val_dataset), limit) if limit else len(val_dataset)
    sums = torch.zeros(8, dtype=torch.float64, device=device)
    local_rows = []
    for index in range(rank, count_total, world_size):
        item = val_dataset[index]
        low = item["LQ"].unsqueeze(0)
        gt = item["GT"].unsqueeze(0)
        low, gt = resize_for_smoke(low, gt, max_edge)
        low, (height, width) = pad_to_multiple(low, int(config["validation"].get("pad_multiple", 16)))
        low = low.to(device=device, dtype=model_dtype)
        gt = gt.to(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(int(config["validation"]["seed"]) + index)
        initial_noise = torch.randn(low.shape, generator=generator, device=device, dtype=torch.float32)
        output = sample_mixed_precision(diffusion, low, initial_noise)[..., :height, :width]
        finite = bool(torch.isfinite(output).all().item())
        sums[6] += 1.0
        if finite:
            raw_psnr, raw_ssim, output_rgb = metric_values(output, gt)
            corrected = match_prediction_mean_to_gt(output, gt)
            corrected_psnr, corrected_ssim, _ = metric_values(corrected, gt)
            lpips_value = float(lpips_alex(output.float(), gt).mean().item()) if lpips_alex else 0.0
            rgb_uint8 = np.rint(output_rgb * 255.0).clip(0, 255).astype(np.uint8)
            gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
            niqe_value = float(niqe(gray))
            values = [raw_psnr, raw_ssim, corrected_psnr, corrected_ssim, lpips_value, niqe_value]
            metrics_finite = all(math.isfinite(value) for value in values)
            finite = finite and metrics_finite
            if finite:
                sums[:6] += torch.tensor(values, dtype=torch.float64, device=device)
                sums[7] += 1.0
                local_rows.append({
                    "index": index,
                    "name": Path(item["LQ_path"]).name,
                    "raw_psnr": raw_psnr,
                    "raw_ssim": raw_ssim,
                    "corrected_psnr": corrected_psnr,
                    "corrected_ssim": corrected_ssim,
                    "lpips_alex": lpips_value,
                    "niqe": niqe_value,
                })
        del item, low, gt, initial_noise, output
        torch.cuda.empty_cache()
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_rows)
        rows = [row for group in gathered for row in group]
    else:
        rows = local_rows
    total = int(sums[6].item())
    finite_count = int(sums[7].item())
    if total != count_total:
        raise RuntimeError(f"Validation accounting mismatch: expected {count_total}, got {total}")
    if finite_count != total:
        raise RuntimeError(f"Non-finite validation output or metric: finite={finite_count}/{total}")
    names = ("raw_psnr", "raw_ssim", "corrected_psnr", "corrected_ssim", "lpips_alex", "niqe")
    result = {name: float(sums[index].item() / total) for index, name in enumerate(names)}
    result.update({
        "count": total,
        "finite_count": finite_count,
        "nfe": int(num_steps),
        "native_resolution": not bool(max_edge),
        "precision": precision,
        "per_image": sorted(rows, key=lambda row: row["index"]),
    })
    del diffusion, model
    torch.cuda.empty_cache()
    return result


def initialize_lpips(device, net="vgg"):
    import lpips

    model = lpips.LPIPS(net=net).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_teacher_validation(config, dataset_config, args, rank, world_size, device):
    if not args.validation_limit and world_size != 8:
        raise RuntimeError(f"Complete teacher ladder requires exactly eight GPU ranks, got {world_size}")
    teacher_payload = load_checkpoint(config["teacher_checkpoint"])
    teacher = build_unet(config)
    teacher_state = extract_network_state(teacher_payload, prefer_ema=True)
    validate_and_load_network(teacher, teacher_state, "teacher checkpoint")
    del teacher
    val_dataset = DLLDataset(dataset_config["datasets"]["val"], train=False)
    lpips_alex = initialize_lpips(device, net="alex")
    run = init_wandb(config, args, rank)
    ladder = sorted(map(int, config["progressive"]["teacher_ladder"]))
    results = []
    for nfe in ladder:
        metrics = evaluate_state(
            config,
            teacher_state,
            val_dataset,
            nfe,
            device,
            rank,
            world_size,
            limit=args.validation_limit,
            max_edge=args.validation_max_edge,
            lpips_alex=lpips_alex,
        )
        results.append(metrics)
        if rank == 0:
            LOGGER.info("Teacher ladder NFE=%d metrics=%s", nfe, json.dumps({k: v for k, v in metrics.items() if k != "per_image"}, sort_keys=True))
            if run:
                run.log({f"teacher_ladder/{key}_nfe_{nfe}": value for key, value in metrics.items() if isinstance(value, (int, float))}, step=0)
    tolerance = float(config["validation"].get("selection_tolerance_db", 1e-8))
    selected = results[0]
    for candidate in results[1:]:
        if candidate["corrected_psnr"] > selected["corrected_psnr"] + tolerance:
            selected = candidate
    teacher_hash = sha256_file(config["teacher_checkpoint"]) if rank == 0 else None
    if world_size > 1:
        values = [teacher_hash]
        dist.broadcast_object_list(values, src=0)
        teacher_hash = values[0]
    result = {
        "passed": all(item["finite_count"] == item["count"] for item in results),
        "selected_nfe": int(selected["nfe"]),
        "selected_by": "corrected_psnr",
        "selection_tolerance_db": tolerance,
        "ladder": results,
        "teacher_checkpoint": str(Path(config["teacher_checkpoint"]).resolve()),
        "teacher_sha256": teacher_hash,
        "source_commit": git_commit(),
        "dataset_count": len(val_dataset),
    }
    if rank == 0:
        output = Path(config["output_dir"]) / "preflight" / "teacher_ladder.json"
        atomic_json_save(result, output)
        LOGGER.info("Selected teacher starting NFE=%d corrected_psnr=%.6f", selected["nfe"], selected["corrected_psnr"])
        if run:
            import wandb

            columns = ["nfe", "raw_psnr", "raw_ssim", "corrected_psnr", "corrected_ssim", "lpips_alex", "niqe"]
            table = wandb.Table(columns=columns)
            for item in results:
                table.add_data(*(item[column] for column in columns))
            run.log({"teacher_ladder/table": table}, step=0)
            run.summary.update({
                "teacher_selected_nfe": int(selected["nfe"]),
                "teacher_selected_corrected_psnr": selected["corrected_psnr"],
                "teacher_sha256": teacher_hash,
                "teacher_ladder_path": str(output.resolve()),
            })
            run.finish()
    barrier(world_size)
    if not result["passed"]:
        raise RuntimeError("Teacher NFE ladder contains non-finite output or metrics")


def init_wandb(config, args, rank):
    if rank != 0 or args.wandb_mode == "disabled":
        return None
    import wandb

    wandb_dir = Path(config["wandb"]["dir"])
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_LOG_MODEL"] = "false"
    run = wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        id=config["wandb"]["id"],
        resume="allow",
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        config=config,
    )
    run.config.update(
        {
            "source_commit": git_commit(),
            "job_id": config.get("job_id", "unknown"),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "teacher_identity": Path(config["teacher_checkpoint"]).name,
        },
        allow_val_change=True,
    )
    return run


def reduce_logs(values, device, world_size):
    keys = sorted(values)
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist())}


def make_scheduler(optimizer, config):
    constant = int(config["train"]["lr_constant_steps"])
    decay = int(config["train"]["lr_decay_steps"])

    def multiplier(step):
        if step <= constant:
            return 1.0
        return max(0.0, 1.0 - (step - constant) / max(decay, 1))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def require_full_gate(config, smoke_steps):
    if smoke_steps:
        return
    gate_path = Path(config["output_dir"]) / "preflight" / "teacher_ladder.json"
    if not gate_path.is_file():
        raise RuntimeError(f"Teacher ladder is missing: {gate_path}")
    with open(gate_path, "r", encoding="utf-8") as handle:
        gate = json.load(handle)
    if not gate.get("passed") or any(item["count"] != 24 for item in gate["ladder"]):
        raise RuntimeError("A finite full 24-image teacher ladder is required")
    return gate


def run_training(config, dataset_config, args, rank, local_rank, world_size, device):
    require_full_gate(config, args.smoke_steps)
    train_dataset = DLLDataset(dataset_config["datasets"]["train"], train=True)
    val_dataset = DLLDataset(dataset_config["datasets"]["val"], train=False)
    global_batch = int(dataset_config["datasets"]["train"]["batch_size"])
    if global_batch % world_size:
        raise ValueError(f"Global batch {global_batch} is not divisible by world size {world_size}")
    per_rank_batch = global_batch // world_size
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )
    loader_options = {
        "dataset": train_dataset,
        "batch_size": per_rank_batch,
        "sampler": sampler,
        "num_workers": int(dataset_config["datasets"]["train"]["n_workers"]),
        "pin_memory": True,
        "drop_last": True,
    }
    if loader_options["num_workers"]:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(**loader_options)
    LOGGER.info(
        "DLL data ready: train=%d val=%d global_batch=%d per_rank_batch=%d steps_per_epoch=%d",
        len(train_dataset),
        len(val_dataset),
        global_batch,
        per_rank_batch,
        len(train_loader),
    )

    lpips_model = initialize_lpips(device)
    run = init_wandb(config, args, rank)
    stages = list(map(int, config["progressive"]["stages"]))
    if any(stages[index] != stages[index + 1] * 2 for index in range(len(stages) - 1)):
        raise ValueError(f"Progressive stages must halve exactly: {stages}")
    master_steps = stages[0]
    stage_iterations = int(config["progressive"]["iterations_per_stage"])
    if args.smoke_steps:
        stage_iterations = int(args.smoke_steps)
        stages = stages[:2]

    resume_path = find_resume_checkpoint(config, args.resume)
    resume_payload = load_checkpoint(resume_path) if resume_path else None
    if resume_payload:
        LOGGER.info("Resuming from %s", resume_path)
    global_step = int(resume_payload.get("global_step", 0)) if resume_payload else 0

    initial_teacher_payload = load_checkpoint(config["teacher_checkpoint"])
    initial_teacher_state = extract_network_state(initial_teacher_payload, prefer_ema=True)

    for stage_index in range(len(stages) - 1):
        teacher_steps, student_steps = stages[stage_index : stage_index + 2]
        if resume_payload and stage_index < int(resume_payload["stage_index"]):
            continue
        if resume_payload and stage_index == int(resume_payload["stage_index"]) and resume_payload.get("stage_complete"):
            continue

        if stage_index == 0:
            source_state = initial_teacher_state
        else:
            previous_teacher, previous_student = stages[stage_index - 1 : stage_index + 1]
            previous_path = stage_directory(config["output_dir"], previous_teacher, previous_student) / "stage_final.pt"
            previous = load_checkpoint(previous_path)
            if not previous.get("stage_complete"):
                raise RuntimeError(f"Previous stage is incomplete: {previous_path}")
            source_state = extract_network_state(previous, prefer_ema=True)

        teacher_network = build_unet(config)
        student_network = build_unet(config)
        validate_and_load_network(teacher_network, source_state, f"{teacher_steps}-step teacher")
        validate_and_load_network(student_network, source_state, f"{student_steps}-step student initialization")
        teacher_network.to(device).eval()
        student_network.to(device).train()
        for parameter in teacher_network.parameters():
            parameter.requires_grad_(False)
        teacher_diffusion = make_diffusion(config, teacher_network, teacher_steps, master_steps).to(device)
        if world_size > 1:
            student_network = DDP(
                student_network,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        optimizer = torch.optim.Adam(student_network.parameters(), lr=float(config["train"]["learning_rate"]))
        scheduler = make_scheduler(optimizer, config)
        ema = ExponentialMovingAverage(student_network, config["train"]["ema_decay"])
        stage_step = 0
        epoch = 0
        batch_in_epoch = 0

        if resume_payload and stage_index == int(resume_payload["stage_index"]):
            validate_and_load_network(
                student_network.module if isinstance(student_network, DDP) else student_network,
                extract_network_state(resume_payload["student"], prefer_ema=False),
                "resumed student",
            )
            ema.load_state_dict(extract_network_state(resume_payload, prefer_ema=True))
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            stage_step = int(resume_payload["stage_step"])
            global_step = int(resume_payload["global_step"])
            epoch = int(resume_payload["epoch"])
            batch_in_epoch = int(resume_payload["batch_in_epoch"])
            restore_rng_state(resume_payload["rng_by_rank"][rank])

        LOGGER.info("Starting phase %d: %d -> %d at stage_step=%d", stage_index, teacher_steps, student_steps, stage_step)
        checkpoint_dir = stage_directory(config["output_dir"], teacher_steps, student_steps)
        latest_path = checkpoint_dir / "latest.pt"

        while stage_step < stage_iterations:
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            completed_epoch = True
            for batch_index, batch in enumerate(train_loader):
                if batch_index < batch_in_epoch:
                    continue
                batch_in_epoch = batch_index
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in ("LQ", "GT")}
                optimizer.zero_grad(set_to_none=True)
                total_loss, parts = teacher_diffusion.loss(
                    batch,
                    student_network,
                    student_steps,
                    lpips_func=lpips_model,
                )
                finite = torch.tensor(float(torch.isfinite(total_loss).item()), device=device)
                if world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise RuntimeError("Non-finite loss detected")
                total_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    student_network.parameters(), float(config["train"]["gradient_clip"])
                )
                optimizer.step()
                scheduler.step()
                ema.update(student_network)
                stage_step += 1
                global_step += 1
                batch_in_epoch = batch_index + 1

                if stage_step % int(config["train"]["log_frequency"]) == 0 or stage_step == 1:
                    values = {
                        "train/total_loss": total_loss.detach(),
                        "train/distill_loss": parts["distill_loss"],
                        "train/pixel_loss": parts["pixel_loss"],
                        "train/perceptual_loss": parts["perceptual_loss"],
                        "train/gradient_norm": gradient_norm,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    values = reduce_logs(values, device, world_size)
                    values.update(
                        {
                            "progress/stage_index": stage_index,
                            "progress/stage_step": stage_step,
                            "progress/epoch": epoch + 1,
                            "progress/teacher_nfe": teacher_steps,
                            "progress/student_nfe": student_steps,
                        }
                    )
                    if rank == 0:
                        LOGGER.info(
                            "phase=%d %d->%d epoch=%d step=%d/%d global=%d loss=%.6f lr=%.3e",
                            stage_index,
                            teacher_steps,
                            student_steps,
                            epoch + 1,
                            stage_step,
                            stage_iterations,
                            global_step,
                            values["train/total_loss"],
                            values["train/learning_rate"],
                        )
                        if run:
                            run.log(values, step=global_step)

                if stage_step % int(config["train"]["checkpoint_frequency"]) == 0:
                    checkpoint_path = save_training_checkpoint(
                        latest_path,
                        config,
                        stage_index,
                        teacher_steps,
                        student_steps,
                        stage_step,
                        global_step,
                        epoch,
                        batch_in_epoch,
                        student_network,
                        ema,
                        optimizer,
                        scheduler,
                        metrics=None,
                        stage_complete=False,
                        rank=rank,
                        world_size=world_size,
                    )
                    if rank == 0 and run:
                        run.summary["latest_checkpoint_path"] = checkpoint_path

                if stage_step >= stage_iterations:
                    completed_epoch = False
                    break

            if completed_epoch:
                epoch += 1
                batch_in_epoch = 0
                if stage_index == 0 and epoch == 2:
                    monitor_path = checkpoint_dir / "epoch_0002.pt"
                    checkpoint_path = save_training_checkpoint(
                        monitor_path,
                        config,
                        stage_index,
                        teacher_steps,
                        student_steps,
                        stage_step,
                        global_step,
                        epoch,
                        0,
                        student_network,
                        ema,
                        optimizer,
                        scheduler,
                        metrics=None,
                        stage_complete=False,
                        rank=rank,
                        world_size=world_size,
                    )
                    if rank == 0 and run:
                        run.summary["epoch_2_checkpoint_path"] = checkpoint_path

        if args.smoke_steps:
            smoke_path = checkpoint_dir / "smoke_final.pt"
            save_training_checkpoint(
                smoke_path,
                config,
                stage_index,
                teacher_steps,
                student_steps,
                stage_step,
                global_step,
                epoch,
                batch_in_epoch,
                student_network,
                ema,
                optimizer,
                scheduler,
                metrics={"smoke": True},
                stage_complete=False,
                rank=rank,
                world_size=world_size,
            )
            break

        del teacher_diffusion, teacher_network
        torch.cuda.empty_cache()
        with ema.apply(student_network) as ema_model:
            metrics = evaluate_model(
                config,
                ema_model,
                val_dataset,
                student_steps,
                master_steps,
                device,
                rank,
                world_size,
                limit=args.validation_limit,
                max_edge=args.validation_max_edge,
                lpips_model=lpips_model,
            )
        stage_final = checkpoint_dir / "stage_final.pt"
        checkpoint_path = save_training_checkpoint(
            stage_final,
            config,
            stage_index,
            teacher_steps,
            student_steps,
            stage_step,
            global_step,
            epoch,
            batch_in_epoch,
            student_network,
            ema,
            optimizer,
            scheduler,
            metrics=metrics,
            stage_complete=True,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            LOGGER.info("Completed phase %d: metrics=%s", stage_index, json.dumps(metrics, sort_keys=True))
            if run:
                run.log(
                    {
                        "validation/psnr": metrics["psnr"],
                        "validation/ssim": metrics["ssim"],
                        "validation/lpips": metrics["lpips"],
                        "progress/student_nfe": student_steps,
                    },
                    step=global_step,
                )
                run.summary[f"stage_{student_steps}_checkpoint_path"] = checkpoint_path
        barrier(world_size)
        del student_network, optimizer, scheduler, ema
        torch.cuda.empty_cache()
        resume_payload = None

    if rank == 0 and run:
        run.summary["training_complete"] = not bool(args.smoke_steps)
        run.finish()


def verify_distributed_shards(sampler, dataset_size, rank, world_size):
    sampler.set_epoch(0)
    local_indices = list(iter(sampler))
    if world_size > 1:
        shards = [None] * world_size
        dist.all_gather_object(shards, local_indices)
    else:
        shards = [local_indices]
    if rank == 0:
        if len(shards) != world_size or any(len(shard) != len(local_indices) for shard in shards):
            raise RuntimeError("Distributed sampler shard sizes are inconsistent")
        if world_size > 1 and len({tuple(shard) for shard in shards}) != world_size:
            raise RuntimeError("Distributed sampler produced identical rank shards")
        unique = len(set().union(*(set(shard) for shard in shards)))
        if unique != dataset_size:
            raise RuntimeError(f"Distributed sampler covers {unique}/{dataset_size} examples")
        LOGGER.info(
            "Verified distinct sampler shards: ranks=%d samples_per_rank=%d unique=%d padding=%d",
            world_size,
            len(local_indices),
            unique,
            sum(map(len, shards)) - unique,
        )
    barrier(world_size)


def stage_transitions(selected_nfe):
    selected_nfe = int(selected_nfe)
    if selected_nfe == 2:
        return [(4, 2)]
    if selected_nfe < 2 or selected_nfe > 512 or selected_nfe & (selected_nfe - 1):
        raise ValueError(f"Selected NFE must be a power of two in [2, 512], got {selected_nfe}")
    values = []
    current = selected_nfe
    while current > 2:
        values.append((current, current // 2))
        current //= 2
    return values


def public_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "per_image"}


def optional_float(value, default):
    return float(default if value is None else value)


def run_progressive_training(config, dataset_config, args, rank, local_rank, world_size, device):
    gate = require_full_gate(config, args.smoke_steps)
    if not args.smoke_steps and world_size != 8:
        raise RuntimeError(f"Long training requires exactly eight GPU ranks, got {world_size}")
    expected_loss = {
        "w_refinement": 0.4,
        "w_pixel": 0.0,
        "w_perceptual": 0.6,
        "w_structural": 0.0,
    }
    if config["model"]["loss"] != expected_loss:
        raise RuntimeError(f"Non-canonical ReDDiT loss configuration: {config['model']['loss']}")
    required_train = {
        "lr_constant_steps": 8000,
        "lr_decay_steps": 12000,
        "ema_decay": 0.9999,
        "gradient_clip": 1.0,
    }
    for key, expected in required_train.items():
        if float(config["train"][key]) != float(expected):
            raise RuntimeError(f"Non-canonical training setting {key}={config['train'][key]}")
    train_options = dataset_config["datasets"]["train"]
    if int(train_options["patch_size"]) != 96 or not train_options["use_crop"] or not train_options["use_flip"] or train_options["use_rot"]:
        raise RuntimeError("DLL augmentation must be paired 96x96 crop plus horizontal flip only")
    train_dataset = DLLDataset(dataset_config["datasets"]["train"], train=True)
    val_dataset = DLLDataset(dataset_config["datasets"]["val"], train=False)
    if len(train_dataset) != 2970 or len(val_dataset) != 24:
        raise RuntimeError(f"Unexpected DLL counts: train={len(train_dataset)} val={len(val_dataset)}")
    global_batch = int(dataset_config["datasets"]["train"]["batch_size"])
    if global_batch != 16 or global_batch % world_size:
        raise ValueError(f"Global batch must be 16 and divisible by world size, got {global_batch}/{world_size}")
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )
    verify_distributed_shards(sampler, len(train_dataset), rank, world_size)
    loader_options = {
        "dataset": train_dataset,
        "batch_size": global_batch // world_size,
        "sampler": sampler,
        "num_workers": int(dataset_config["datasets"]["train"]["n_workers"]),
        "pin_memory": True,
        "drop_last": True,
    }
    if loader_options["num_workers"]:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(**loader_options)
    LOGGER.info(
        "DLL ready: train=%d val=%d global_batch=%d per_rank_batch=%d steps_per_epoch=%d",
        len(train_dataset), len(val_dataset), global_batch, global_batch // world_size, len(train_loader),
    )

    initial_payload = load_checkpoint(config["teacher_checkpoint"])
    initial_state = extract_network_state(initial_payload, prefer_ema=True)
    strict_probe = build_unet(config)
    validate_and_load_network(strict_probe, initial_state, "initial teacher")
    del strict_probe, initial_payload
    selected_nfe = int(gate["selected_nfe"]) if gate else max(config["progressive"]["teacher_ladder"])
    transitions = stage_transitions(selected_nfe)
    if args.smoke_steps:
        transitions = transitions[:1]
    stage_iterations = int(args.smoke_steps or config["progressive"]["iterations_per_stage"])
    master_steps = int(config["model"]["schedule"]["master_n_timestep"])
    if master_steps != 512:
        raise RuntimeError(f"Canonical training requires a 512-interval shared schedule, got {master_steps}")

    lpips_vgg = initialize_lpips(device, net="vgg")
    lpips_alex = initialize_lpips(device, net="alex")
    run = init_wandb(config, args, rank)
    resume_path = find_resume_checkpoint(config, args.resume)
    resume_payload = load_checkpoint(resume_path) if resume_path else None
    if resume_payload:
        if resume_payload.get("kind") != "reddit_progressive_distillation":
            raise RuntimeError(f"Refusing incompatible resume checkpoint: {resume_path}")
        LOGGER.info("Resume candidate: %s stage=%s complete=%s", resume_path, resume_payload["stage_index"], resume_payload["stage_complete"])
    global_step = int(resume_payload.get("global_step", 0)) if resume_payload else 0
    manifest_path = Path(config["output_dir"]) / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    global_best_2 = optional_float(manifest.get("best_2nfe_raw_psnr"), "-inf")
    restart_count = int(manifest.get("restart_count", 0))
    if args.restart_reason:
        restart_count += 1
        if rank == 0:
            reasons = list(manifest.get("restart_reasons", []))
            reasons.append({"reason": args.restart_reason, "learning_rate": config["train"]["learning_rate"], "time": time.time()})
            update_manifest(config, restart_count=restart_count, restart_reasons=reasons)
            if run:
                run.log({"restart/count": restart_count, "restart/reason": args.restart_reason}, step=global_step)
    barrier(world_size)

    train_precision = str(config["train"].get("precision", "bfloat16"))
    use_autocast = train_precision == "bfloat16"
    if train_precision not in ("bfloat16", "float32"):
        raise ValueError(f"Unsupported training precision: {train_precision}")
    best_four_state = initial_state if selected_nfe <= 4 else None

    def run_stage(stage_index, teacher_nfe, student_nfe, teacher_state, student_state, extension_cycle, learning_rate, resume):
        nonlocal global_step, global_best_2
        phase_kind = "extension" if extension_cycle else "progressive"
        checkpoint_dir = stage_directory(config["output_dir"], teacher_nfe, student_nfe, extension_cycle)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = Path(config["output_dir"]) / "latest.pt"
        best_path = checkpoint_dir / "best_raw_psnr.pt"

        teacher_network = build_unet(config)
        student_network = build_unet(config)
        validate_and_load_network(teacher_network, teacher_state, f"{teacher_nfe}-NFE frozen teacher")
        validate_and_load_network(student_network, student_state, f"{student_nfe}-NFE student initialization")
        teacher_network.to(device).eval()
        student_network.to(device).train()
        for parameter in teacher_network.parameters():
            parameter.requires_grad_(False)
        teacher_diffusion = make_diffusion(config, teacher_network, teacher_nfe, master_steps).to(device)
        if world_size > 1:
            student_network = DDP(
                student_network,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        optimizer = torch.optim.Adam(student_network.parameters(), lr=float(learning_rate))
        scheduler = make_scheduler(optimizer, config)
        ema = ExponentialMovingAverage(student_network, config["train"]["ema_decay"])
        stage_step = 0
        epoch = 0
        batch_in_epoch = 0
        stage_best = float("-inf")
        if best_path.is_file():
            prior_best = load_checkpoint(best_path)
            stage_best = float(prior_best.get("metrics", {}).get("raw_psnr", "-inf"))
            del prior_best

        if resume and int(resume["stage_index"]) == stage_index and not resume.get("stage_complete"):
            if int(resume["teacher_steps"]) != teacher_nfe or int(resume["student_steps"]) != student_nfe:
                raise RuntimeError("Resume phase does not match the requested phase")
            validate_and_load_network(teacher_network, extract_network_state(resume["teacher"], prefer_ema=False), "resumed teacher")
            validate_and_load_network(
                student_network.module if isinstance(student_network, DDP) else student_network,
                extract_network_state(resume["student"], prefer_ema=False),
                "resumed student",
            )
            ema.load_state_dict(extract_network_state(resume, prefer_ema=True))
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            stage_step = int(resume["stage_step"])
            global_step = int(resume["global_step"])
            epoch = int(resume["epoch"])
            batch_in_epoch = int(resume["batch_in_epoch"])
            if len(resume["rng_by_rank"]) != world_size:
                raise RuntimeError("Resume world size differs from checkpoint world size")
            restore_rng_state(resume["rng_by_rank"][rank])
            LOGGER.info("Exact resume %s %d->%d cycle=%d epoch=%d batch=%d step=%d", phase_kind, teacher_nfe, student_nfe, extension_cycle, epoch, batch_in_epoch, stage_step)
        else:
            LOGGER.info("Start %s %d->%d cycle=%d lr=%.3e steps=%d", phase_kind, teacher_nfe, student_nfe, extension_cycle, learning_rate, stage_iterations)

        metadata = {
            "phase_kind": phase_kind,
            "extension_cycle": int(extension_cycle),
            "learning_rate": float(learning_rate),
            "sampler_epoch": int(epoch),
            "restart_count": restart_count,
        }

        def save_current(path, metrics, complete=False):
            return save_training_checkpoint(
                path, config, stage_index, teacher_nfe, student_nfe, stage_step, global_step,
                epoch, batch_in_epoch, student_network, ema, optimizer, scheduler, metrics,
                complete, rank, world_size, teacher=teacher_network, metadata=metadata,
            )

        def validate_ema(reason):
            nonlocal stage_best, global_best_2
            metrics = evaluate_state(
                config, ema.state_dict_cpu(), val_dataset, student_nfe, device, rank, world_size,
                limit=args.validation_limit, max_edge=args.validation_max_edge, lpips_alex=lpips_alex,
            )
            current = float(metrics["raw_psnr"])
            improved_stage = current > stage_best
            improved_two = student_nfe == 2 and current > global_best_2
            if improved_stage:
                stage_best = current
                save_current(best_path, metrics)
            if improved_two:
                global_best_2 = current
                if rank == 0:
                    atomic_copy(best_path, Path(config["output_dir"]) / "best_2nfe.pt")
                barrier(world_size)
            checkpoint_path = save_current(latest_path, metrics)
            if rank == 0:
                values = {f"validation/{key}": value for key, value in public_metrics(metrics).items() if isinstance(value, (int, float))}
                values.update({
                    "validation/reason": reason,
                    "validation/best_stage_raw_psnr": stage_best,
                    "validation/best_2nfe_raw_psnr": global_best_2 if math.isfinite(global_best_2) else None,
                    "progress/stage_index": stage_index,
                    "progress/stage_step": stage_step,
                    "progress/epoch": epoch,
                    "progress/teacher_nfe": teacher_nfe,
                    "progress/student_nfe": student_nfe,
                    "progress/extension_cycle": extension_cycle,
                })
                LOGGER.info("Validation %s %d->%d epoch=%d step=%d: %s", reason, teacher_nfe, student_nfe, epoch, stage_step, json.dumps(public_metrics(metrics), sort_keys=True))
                update_manifest(
                    config,
                    selected_teacher_nfe=selected_nfe,
                    stage_index=stage_index,
                    phase_kind=phase_kind,
                    extension_cycle=extension_cycle,
                    teacher_nfe=teacher_nfe,
                    student_nfe=student_nfe,
                    epoch=epoch,
                    stage_step=stage_step,
                    global_step=global_step,
                    latest_checkpoint=checkpoint_path,
                    stage_best_checkpoint=str(best_path.resolve()),
                    best_2nfe_checkpoint=str((Path(config["output_dir"]) / "best_2nfe.pt").resolve()) if math.isfinite(global_best_2) else None,
                    best_2nfe_raw_psnr=global_best_2 if math.isfinite(global_best_2) else None,
                    latest_metrics=public_metrics(metrics),
                    restart_count=restart_count,
                )
                if run:
                    run.log(values, step=global_step)
                    run.summary.update({
                        "latest_checkpoint_path": checkpoint_path,
                        "stage_best_checkpoint_path": str(best_path.resolve()),
                        "best_2nfe_checkpoint_path": str((Path(config["output_dir"]) / "best_2nfe.pt").resolve()) if math.isfinite(global_best_2) else "",
                        "best_2nfe_raw_psnr": global_best_2 if math.isfinite(global_best_2) else None,
                    })
            barrier(world_size)
            return metrics

        if stage_step == 0 and not best_path.is_file():
            validate_ema("stage_baseline")

        while stage_step < stage_iterations:
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            reached_target = False
            for batch_index, batch in enumerate(train_loader):
                if batch_index < batch_in_epoch:
                    continue
                batch_in_epoch = batch_index
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in ("LQ", "GT")}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    total_loss, parts = teacher_diffusion.loss(batch, student_network, student_nfe, lpips_func=lpips_vgg)
                finite_loss = torch.tensor(int(torch.isfinite(total_loss).item()), device=device)
                if world_size > 1:
                    dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
                if not finite_loss.item():
                    raise RuntimeError(f"Non-finite loss at stage step {stage_step}")
                total_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(student_network.parameters(), float(config["train"]["gradient_clip"]))
                finite_gradient = torch.tensor(int(torch.isfinite(gradient_norm).item()), device=device)
                if world_size > 1:
                    dist.all_reduce(finite_gradient, op=dist.ReduceOp.MIN)
                if not finite_gradient.item():
                    raise RuntimeError(f"Non-finite gradient at stage step {stage_step}")
                optimizer.step()
                scheduler.step()
                ema.update(student_network)
                stage_step += 1
                global_step += 1
                batch_in_epoch = batch_index + 1
                metadata["sampler_epoch"] = epoch

                if stage_step == 1 or stage_step % int(config["train"]["log_frequency"]) == 0:
                    values = {
                        "train/total_loss": total_loss.detach(),
                        "train/distill_loss": parts["distill_loss"],
                        "train/loss_x0": parts["loss_x0"],
                        "train/loss_epsilon": parts["loss_epsilon"],
                        "train/pixel_loss": parts["pixel_loss"],
                        "train/perceptual_loss": parts["perceptual_loss"],
                        "train/gradient_norm": gradient_norm,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    values = reduce_logs(values, device, world_size)
                    values.update({
                        "progress/stage_index": stage_index,
                        "progress/stage_step": stage_step,
                        "progress/epoch": epoch + 1,
                        "progress/teacher_nfe": teacher_nfe,
                        "progress/student_nfe": student_nfe,
                        "progress/extension_cycle": extension_cycle,
                    })
                    if rank == 0:
                        LOGGER.info("%s %d->%d cycle=%d epoch=%d step=%d/%d loss=%.6f lr=%.3e", phase_kind, teacher_nfe, student_nfe, extension_cycle, epoch + 1, stage_step, stage_iterations, values["train/total_loss"], values["train/learning_rate"])
                        if run:
                            run.log(values, step=global_step)

                if stage_step % int(config["train"]["checkpoint_frequency"]) == 0:
                    save_current(latest_path, None)
                if stage_step >= stage_iterations:
                    reached_target = True
                    break

            completed_epoch = batch_in_epoch >= len(train_loader)
            if completed_epoch:
                epoch += 1
                batch_in_epoch = 0
                validate_ema("epoch")
            if reached_target:
                if not completed_epoch:
                    validate_ema("stage_end")
                break

        if args.smoke_steps:
            smoke_path = checkpoint_dir / "smoke_final.pt"
            save_current(smoke_path, {"smoke": True})
            return str(smoke_path), ema.state_dict_cpu(), stage_best

        if not best_path.is_file():
            raise RuntimeError(f"Stage has no validated best checkpoint: {best_path}")
        best_payload = load_checkpoint(best_path)
        best_payload["stage_complete"] = True
        best_payload["metadata"] = dict(best_payload.get("metadata", {}), completed_at=time.time())
        stage_final = checkpoint_dir / "stage_final.pt"
        if rank == 0:
            atomic_torch_save(best_payload, stage_final)
            atomic_json_save({"checkpoint": str(stage_final.resolve())}, Path(config["output_dir"]) / "latest.json")
        barrier(world_size)
        final_state = extract_network_state(best_payload, prefer_ema=True)
        final_metric = float(best_payload["metrics"]["raw_psnr"])
        if rank == 0:
            update_manifest(config, stage_complete=True, stage_final_checkpoint=str(stage_final.resolve()), stage_best_raw_psnr=final_metric)
            if run:
                run.summary[f"stage_{stage_index}_final_checkpoint_path"] = str(stage_final.resolve())
        barrier(world_size)
        del best_payload, teacher_diffusion, teacher_network, student_network, optimizer, scheduler, ema
        torch.cuda.empty_cache()
        return str(stage_final), final_state, final_metric

    next_source = initial_state
    resume_stage = int(resume_payload["stage_index"]) if resume_payload else 0
    for stage_index, (teacher_nfe, student_nfe) in enumerate(transitions):
        stage_final = stage_directory(config["output_dir"], teacher_nfe, student_nfe) / "stage_final.pt"
        if resume_payload and stage_index < resume_stage:
            payload = load_checkpoint(stage_final)
            if not payload.get("stage_complete"):
                raise RuntimeError(f"Skipped stage is incomplete: {stage_final}")
            next_source = extract_network_state(payload, prefer_ema=True)
            if student_nfe == 4:
                best_four_state = next_source
            del payload
            continue
        if resume_payload and stage_index == resume_stage and resume_payload.get("stage_complete"):
            next_source = extract_network_state(resume_payload, prefer_ema=True)
            if student_nfe == 4:
                best_four_state = next_source
            resume_payload = None
            continue
        _, next_source, _ = run_stage(
            stage_index, teacher_nfe, student_nfe, next_source, next_source, 0,
            float(config["train"]["learning_rate"]), resume_payload if stage_index == resume_stage else None,
        )
        if student_nfe == 4:
            best_four_state = next_source
        resume_payload = None

    if args.smoke_steps or args.stop_after_stage:
        if rank == 0 and run:
            run.summary["smoke_complete"] = bool(args.smoke_steps)
            run.finish()
        return
    if best_four_state is None:
        raise RuntimeError("A frozen 4-NFE teacher state is unavailable for 2-NFE extensions")
    best_two_path = Path(config["output_dir"]) / "best_2nfe.pt"
    if not best_two_path.is_file():
        raise RuntimeError(f"2-NFE best checkpoint missing after progressive stages: {best_two_path}")

    resume_cycle = 0
    if resume_payload and resume_payload.get("metadata", {}).get("phase_kind") == "extension":
        resume_cycle = int(resume_payload["metadata"]["extension_cycle"])
    elif resume_path and resume_payload is None:
        pointer_payload = load_checkpoint(resume_path)
        if pointer_payload.get("metadata", {}).get("phase_kind") == "extension":
            resume_payload = pointer_payload
            resume_cycle = int(pointer_payload["metadata"]["extension_cycle"])
        else:
            del pointer_payload
    extension_cycle = max(1, resume_cycle)
    extension_lr = float(config["train"]["learning_rate"])
    if resume_payload and resume_cycle:
        extension_lr = float(resume_payload.get("metadata", {}).get("learning_rate", extension_lr))
        if resume_payload.get("stage_complete"):
            extension_cycle += 1
            resume_payload = None
    lower_lrs = [float(value) for value in config["train"]["restart_learning_rates"]]
    while True:
        before = global_best_2
        best_two_payload = load_checkpoint(best_two_path)
        student_state = extract_network_state(best_two_payload, prefer_ema=True)
        del best_two_payload
        stage_index = len(transitions) + extension_cycle - 1
        _, _, _ = run_stage(
            stage_index, 4, 2, best_four_state, student_state, extension_cycle,
            extension_lr, resume_payload,
        )
        resume_payload = None
        improvement = global_best_2 - before
        if improvement < float(config["train"]["plateau_min_improvement_db"]):
            candidates = [value for value in lower_lrs if value < extension_lr]
            if candidates:
                extension_lr = max(candidates)
        if rank == 0:
            update_manifest(config, last_extension_improvement_db=improvement, next_extension_learning_rate=extension_lr)
        barrier(world_size)
        extension_cycle += 1


def main():
    args = parse_args()
    config, dataset_config = load_configs(args)
    rank, local_rank, world_size, device = setup_distributed()
    configure_logging(rank, config["output_dir"])
    seed = int(config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    LOGGER.info("mode=%s world_size=%d local_rank=%d commit=%s", args.mode, world_size, local_rank, git_commit())
    try:
        if args.mode == "validate-teacher":
            run_teacher_validation(config, dataset_config, args, rank, world_size, device)
        else:
            run_progressive_training(config, dataset_config, args, rank, local_rank, world_size, device)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
