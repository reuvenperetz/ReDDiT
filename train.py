"""DLL progressive ReDDiT training with DDP, W&B, and exact resume."""

import argparse
import contextlib
import copy
import json
import logging
import math
import os
import random
import subprocess
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


LOGGER = logging.getLogger("reddit")
NETWORK_PREFIXES = ("module.", "model.", "netG.", "generator.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dll_train.json")
    parser.add_argument("--dataset", default="config/dll.yml")
    parser.add_argument("--mode", choices=("train", "validate-teacher"), default="train")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", default="auto", help="Checkpoint path, auto, or none")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--validation-max-edge", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
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


def stage_directory(output_dir, teacher_steps, student_steps):
    return Path(output_dir) / "checkpoints" / f"phase_{teacher_steps}_to_{student_steps}"


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
):
    rng = capture_rng_state()
    if world_size > 1:
        rng_by_rank = [None] * world_size
        dist.all_gather_object(rng_by_rank, rng)
    else:
        rng_by_rank = [rng]

    if rank == 0:
        payload = {
            "format_version": 1,
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
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_by_rank": rng_by_rank,
            "metrics": metrics or {},
            "config": config,
        }
        atomic_torch_save(payload, path)
        atomic_json_save({"checkpoint": str(Path(path).resolve())}, Path(config["output_dir"]) / "latest.json")
        LOGGER.info("Saved checkpoint: %s", path)
    barrier(world_size)
    return str(Path(path).resolve())


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


def resize_pair(item, width, height):
    def tensor_to_rgb(tensor):
        return np.clip(((tensor.numpy().transpose(1, 2, 0) + 1.0) * 0.5), 0.0, 1.0)

    low = tensor_to_rgb(item["LQ"])
    gt = tensor_to_rgb(item["GT"])
    low = cv2.resize(low, (width, height), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_AREA)
    low_tensor = torch.from_numpy(low.transpose(2, 0, 1)).float().mul_(2).sub_(1).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).float().mul_(2).sub_(1).unsqueeze(0)
    return low_tensor, gt_tensor


def metric_values(output, ground_truth):
    output = np.clip((output.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0) + 1) * 0.5, 0, 1)
    ground_truth = np.clip(
        (ground_truth.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0) + 1) * 0.5,
        0,
        1,
    )
    mse = float(np.mean((output - ground_truth) ** 2))
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    ssim = float(structural_similarity(ground_truth, output, data_range=1.0, channel_axis=2))
    return psnr, ssim


@torch.no_grad()
def evaluate_model(
    config,
    model,
    val_dataset,
    num_steps,
    master_steps,
    device,
    rank,
    world_size,
    limit=0,
    max_edge=0,
    lpips_model=None,
):
    validation = config["validation"]
    width = int(validation["width"])
    height = int(validation["height"])
    if max_edge and max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        width = max(16, int(round(width * scale / 16)) * 16)
        height = max(16, int(round(height * scale / 16)) * 16)

    diffusion = make_diffusion(config, model, num_steps, master_steps).to(device)
    diffusion.eval()
    count_total = min(len(val_dataset), limit) if limit else len(val_dataset)
    sums = torch.zeros(5, dtype=torch.float64, device=device)
    for index in range(rank, count_total, world_size):
        low, gt = resize_pair(val_dataset[index], width, height)
        low, gt = low.to(device), gt.to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(validation["seed"]) + index)
        initial_noise = torch.randn(low.shape, device=device, generator=generator)
        output = diffusion.super_resolution(low, initial_noise=initial_noise)
        finite = bool(torch.isfinite(output).all().item())
        if finite:
            psnr, ssim = metric_values(output, gt)
            lpips_value = float(lpips_model(output, gt).mean().item()) if lpips_model else 0.0
            sums += torch.tensor([psnr, ssim, lpips_value, 1.0, 1.0], dtype=torch.float64, device=device)
        else:
            sums[3] += 1.0
        del low, gt, output, initial_noise
        torch.cuda.empty_cache()
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    total = int(sums[3].item())
    finite_count = int(sums[4].item())
    if total != count_total:
        raise RuntimeError(f"Validation accounting mismatch: expected {count_total}, got {total}")
    return {
        "psnr": float(sums[0].item() / max(finite_count, 1)),
        "ssim": float(sums[1].item() / max(finite_count, 1)),
        "lpips": float(sums[2].item() / max(finite_count, 1)) if lpips_model else None,
        "count": total,
        "finite_count": finite_count,
        "width": width,
        "height": height,
        "nfe": int(num_steps),
    }


def initialize_lpips(device):
    import lpips

    model = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_teacher_validation(config, dataset_config, args, rank, world_size, device):
    teacher_payload = load_checkpoint(config["teacher_checkpoint"])
    teacher = build_unet(config)
    validate_and_load_network(teacher, extract_network_state(teacher_payload, prefer_ema=True), "teacher EMA")
    teacher.to(device).eval()
    val_dataset = DLLDataset(dataset_config["datasets"]["val"], train=False)
    baseline_nfe = int(config["validation"]["teacher_baseline_nfe"])
    candidate_nfe = int(config["validation"]["teacher_candidate_nfe"])
    baseline = evaluate_model(
        config,
        teacher,
        val_dataset,
        baseline_nfe,
        baseline_nfe,
        device,
        rank,
        world_size,
        limit=args.validation_limit,
        max_edge=args.validation_max_edge,
    )
    candidate = evaluate_model(
        config,
        teacher,
        val_dataset,
        candidate_nfe,
        candidate_nfe,
        device,
        rank,
        world_size,
        limit=args.validation_limit,
        max_edge=args.validation_max_edge,
    )
    allowed_drop = float(config["validation"]["max_psnr_drop_db"])
    passed = (
        baseline["finite_count"] == baseline["count"]
        and candidate["finite_count"] == candidate["count"]
        and candidate["psnr"] >= baseline["psnr"] - allowed_drop
    )
    result = {
        "passed": bool(passed),
        "allowed_psnr_drop_db": allowed_drop,
        "observed_psnr_drop_db": baseline["psnr"] - candidate["psnr"],
        "baseline": baseline,
        "candidate": candidate,
        "teacher_checkpoint": config["teacher_checkpoint"],
        "teacher_s3_uri": config["teacher_s3_uri"],
        "source_commit": git_commit(),
    }
    if rank == 0:
        output = Path(config["output_dir"]) / "preflight" / "teacher_schedule_gate.json"
        atomic_json_save(result, output)
        LOGGER.info("Teacher schedule gate: %s", json.dumps(result, sort_keys=True))
    barrier(world_size)
    if not passed:
        raise RuntimeError("512-step teacher schedule failed the configured PSNR gate")


def init_wandb(config, args, rank):
    if rank != 0 or args.wandb_mode == "disabled":
        return None
    import wandb

    wandb_dir = Path(config["wandb"]["dir"])
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_LOG_MODEL"] = "false"
    return wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        id=config["wandb"]["id"],
        resume="allow",
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        config=config,
    )


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
    gate_path = Path(config["output_dir"]) / "preflight" / "teacher_schedule_gate.json"
    if not gate_path.is_file():
        raise RuntimeError(f"Teacher schedule gate is missing: {gate_path}")
    with open(gate_path, "r", encoding="utf-8") as handle:
        gate = json.load(handle)
    if not gate.get("passed") or gate["baseline"]["count"] != 24:
        raise RuntimeError("A passing full 24-image teacher schedule gate is required")


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
    LOGGER.info("mode=%s world_size=%d local_rank=%d commit=%s", args.mode, world_size, local_rank, git_commit())
    try:
        if args.mode == "validate-teacher":
            run_teacher_validation(config, dataset_config, args, rank, world_size, device)
        else:
            run_training(config, dataset_config, args, rank, local_rank, world_size, device)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
