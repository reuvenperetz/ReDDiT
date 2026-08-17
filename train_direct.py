"""Direct two-step DLL trajectory fine-tuning with native-resolution validation."""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import train as core
from data.DLL_dataset import DLLDataset


LOGGER = logging.getLogger("reddit.direct")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dll_direct_2step.json")
    parser.add_argument("--dataset", default="config/dll.yml")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-root")
    parser.add_argument("--val-root")
    parser.add_argument("--resume", default="auto", help="Checkpoint path, auto, or none")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--validation-max-edge", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-name", default="Reddit-2step")
    parser.add_argument("--wandb-id")
    return parser.parse_args()


def load_configs(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    with open(args.dataset, "r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)
    if dataset_config.get("dataset") != "DLL":
        raise ValueError("Direct two-step training supports DLL only")
    if args.train_root:
        dataset_config["datasets"]["train"]["root"] = args.train_root
    if args.val_root:
        dataset_config["datasets"]["val"]["root"] = args.val_root
    config["teacher_checkpoint"] = str(Path(args.teacher_checkpoint).resolve())
    config["output_dir"] = str(Path(args.output_dir).resolve())
    return config, dataset_config


def pad_to_multiple(tensor, multiple=16):
    height, width = tensor.shape[-2:]
    pad_h = (-height) % int(multiple)
    pad_w = (-width) % int(multiple)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (height, width)


def resize_for_smoke(low, gt, max_edge):
    height, width = low.shape[-2:]
    if not max_edge or max(height, width) <= max_edge:
        return low, gt
    scale = float(max_edge) / max(height, width)
    target_h = max(16, int(round(height * scale)))
    target_w = max(16, int(round(width * scale)))
    low = F.interpolate(low, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True)
    gt = F.interpolate(gt, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True)
    return low, gt


def match_prediction_mean_to_gt(output, ground_truth):
    prediction = ((output.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    target = ((ground_truth.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    prediction_mean = prediction.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    target_mean = target.mean(dim=(1, 2, 3), keepdim=True)
    return (prediction * (target_mean / prediction_mean)).clamp(0.0, 1.0).mul(2.0).sub(1.0)


@torch.no_grad()
def evaluate_state(config, state, dataset, device, rank, world_size, limit=0, max_edge=0):
    model = core.build_unet(config)
    core.validate_and_load_network(model, state, "validation student")
    precision = config["validation"].get("precision", "bfloat16")
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    student_nfe = int(config["direct"]["student_nfe"])
    master_steps = int(config["model"]["schedule"]["master_n_timestep"])
    diffusion = core.make_diffusion(config, model, student_nfe, master_steps).to(device=device, dtype=dtype).eval()
    count_total = min(len(dataset), limit) if limit else len(dataset)
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    for index in range(rank, count_total, world_size):
        item = dataset[index]
        low = item["LQ"].unsqueeze(0)
        gt = item["GT"].unsqueeze(0)
        low, gt = resize_for_smoke(low, gt, max_edge)
        low, (height, width) = pad_to_multiple(low, 16)
        low = low.to(device=device, dtype=dtype)
        gt = gt.to(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(int(config["validation"]["seed"]) + index)
        noise = torch.randn(low.shape, device=device, generator=generator, dtype=torch.float32).to(dtype)
        output = diffusion.super_resolution(low, initial_noise=noise)[..., :height, :width].float()
        finite = bool(torch.isfinite(output).all().item())
        if finite:
            raw_psnr, raw_ssim = core.metric_values(output, gt)
            matched = match_prediction_mean_to_gt(output, gt)
            psnr, ssim = core.metric_values(matched, gt)
            sums += torch.tensor(
                [raw_psnr, raw_ssim, psnr, ssim, 1.0, 1.0],
                dtype=torch.float64,
                device=device,
            )
        else:
            sums[4] += 1.0
        del item, low, gt, noise, output
        if finite:
            del matched
        torch.cuda.empty_cache()
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    total = int(sums[4].item())
    finite_count = int(sums[5].item())
    if total != count_total:
        raise RuntimeError(f"Validation accounting mismatch: expected {count_total}, got {total}")
    metrics = {
        "psnr": float(sums[2].item() / max(finite_count, 1)),
        "ssim": float(sums[3].item() / max(finite_count, 1)),
        "raw_psnr": float(sums[0].item() / max(finite_count, 1)),
        "raw_ssim": float(sums[1].item() / max(finite_count, 1)),
        "count": total,
        "finite_count": finite_count,
        "nfe": student_nfe,
        "native_resolution": not bool(max_edge),
        "mean_matched": bool(config["validation"].get("mean_match", True)),
    }
    del diffusion, model
    torch.cuda.empty_cache()
    return metrics


def direct_trajectory_loss(diffusion, batch, lpips_model, config):
    low = batch["LQ"]
    target = batch["GT"]
    x_t = torch.randn_like(target)
    intermediate = []
    for step in range(diffusion.num_timesteps, 0, -1):
        t_index = step * diffusion.time_scale
        s_index = (step - 1) * diffusion.time_scale
        alpha_t = diffusion.sqrt_alphas_cumprod[t_index].expand(target.shape[0], 1)
        eps, _ = diffusion.denoise_fn(torch.cat([low, x_t], dim=1), alpha_t)
        x_0 = diffusion.predict_start(x_t, diffusion._expand(alpha_t[:, 0]), eps)
        intermediate.append(x_0)
        clipped = x_0.clamp(-1.0, 1.0)
        eps = diffusion.predict_eps(x_t, clipped, diffusion._expand(alpha_t[:, 0]))
        alpha_s = diffusion.sqrt_alphas_cumprod[s_index]
        sigma_s = diffusion.sqrt_one_minus_alphas_cumprod[s_index]
        x_t = alpha_s * clipped + sigma_s * eps
    output = x_t.clamp(-1.0, 1.0)
    l1 = F.l1_loss(output, target)
    mse = F.mse_loss(output, target)
    perceptual = lpips_model(output, target).mean()
    weights = config["direct"]
    loss = (
        float(weights["pixel_l1_weight"]) * l1
        + float(weights["pixel_mse_weight"]) * mse
        + float(weights["perceptual_weight"]) * perceptual
    )
    return loss, {"l1": l1.detach(), "mse": mse.detach(), "perceptual": perceptual.detach()}


def init_wandb(config, args, rank):
    if rank != 0 or args.wandb_mode == "disabled":
        return None
    import wandb

    wandb_dir = Path(config["output_dir"]) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_LOG_MODEL"] = "false"
    kwargs = {
        "project": config["wandb"]["project"],
        "name": args.wandb_name,
        "mode": args.wandb_mode,
        "dir": str(wandb_dir),
        "config": config,
    }
    if args.wandb_id:
        kwargs.update(id=args.wandb_id, resume="allow")
    return wandb.init(**kwargs)


def checkpoint_payload(config, step, epoch, student, ema, optimizer, metrics, rank, world_size):
    rng = core.capture_rng_state()
    if world_size > 1:
        rng_by_rank = [None] * world_size
        dist.all_gather_object(rng_by_rank, rng)
    else:
        rng_by_rank = [rng]
    if rank != 0:
        return None
    return {
        "format_version": 1,
        "kind": "reddit_direct_2step",
        "source_commit": core.git_commit(),
        "step": int(step),
        "epoch": int(epoch),
        "student_nfe": int(config["direct"]["student_nfe"]),
        "student": core.state_dict_cpu(student),
        "ema": ema.state_dict_cpu(),
        "optimizer": optimizer.state_dict(),
        "rng_by_rank": rng_by_rank,
        "metrics": metrics or {},
        "config": config,
    }


def save_checkpoint(path, config, step, epoch, student, ema, optimizer, metrics, rank, world_size):
    payload = checkpoint_payload(config, step, epoch, student, ema, optimizer, metrics, rank, world_size)
    if rank == 0:
        core.atomic_torch_save(payload, path)
        LOGGER.info("Saved checkpoint: %s", Path(path).resolve())
    core.barrier(world_size)


def find_resume(output_dir, requested):
    if requested.lower() in ("none", "false", "off"):
        return None
    if requested != "auto":
        return Path(requested)
    candidate = Path(output_dir) / "checkpoints" / "latest.pt"
    return candidate if candidate.is_file() else None


def run_training(config, dataset_config, args, rank, local_rank, world_size, device):
    train_set = DLLDataset(dataset_config["datasets"]["train"], train=True)
    val_set = DLLDataset(dataset_config["datasets"]["val"], train=False)
    global_batch = int(dataset_config["datasets"]["train"]["batch_size"])
    if global_batch % world_size:
        raise ValueError(f"Global batch {global_batch} is not divisible by world size {world_size}")
    sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )
    workers = int(dataset_config["datasets"]["train"]["n_workers"])
    loader_kwargs = {
        "dataset": train_set,
        "batch_size": global_batch // world_size,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if workers:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    teacher_payload = core.load_checkpoint(config["teacher_checkpoint"])
    teacher_state = core.extract_network_state(teacher_payload, prefer_ema=True)
    student = core.build_unet(config)
    core.validate_and_load_network(student, teacher_state, "teacher EMA student initialization")
    student.to(device).train()
    if world_size > 1:
        student = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config["direct"]["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    ema = core.ExponentialMovingAverage(student, float(config["direct"]["ema_decay"]))
    lpips_model = core.initialize_lpips(device)
    master_steps = int(config["model"]["schedule"]["master_n_timestep"])
    diffusion = core.make_diffusion(
        config,
        student,
        int(config["direct"]["student_nfe"]),
        master_steps,
    ).to(device)
    run = init_wandb(config, args, rank)

    checkpoint_dir = Path(config["output_dir"]) / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    core.barrier(world_size)
    resume_path = find_resume(config["output_dir"], args.resume)
    step = 0
    epoch = 0
    best_metrics = None
    if resume_path:
        payload = core.load_checkpoint(resume_path)
        core.validate_and_load_network(
            student.module if isinstance(student, DDP) else student,
            core.extract_network_state(payload["student"], prefer_ema=False),
            "resumed student",
        )
        ema.load_state_dict(core.extract_network_state(payload, prefer_ema=True))
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        epoch = int(payload["epoch"])
        core.restore_rng_state(payload["rng_by_rank"][rank])
        best_path = checkpoint_dir / "best.pt"
        if best_path.is_file():
            best_metrics = core.load_checkpoint(best_path).get("metrics")
        LOGGER.info("Resumed direct training from %s at step=%d", resume_path, step)

    smoke = int(args.smoke_steps)
    iterations = smoke or int(config["direct"]["iterations"])
    minimum_iterations = iterations if smoke else int(config["direct"]["minimum_iterations"])
    validation_frequency = iterations if smoke else int(config["direct"]["validation_frequency"])
    validation_limit = args.validation_limit
    validation_max_edge = args.validation_max_edge
    target_psnr = float(config["validation"]["target_psnr"])

    if step == 0:
        initial_metrics = evaluate_state(
            config,
            ema.state_dict_cpu(),
            val_set,
            device,
            rank,
            world_size,
            limit=validation_limit,
            max_edge=validation_max_edge,
        )
        best_metrics = initial_metrics
        save_checkpoint(
            checkpoint_dir / "best.pt",
            config,
            step,
            epoch,
            student,
            ema,
            optimizer,
            initial_metrics,
            rank,
            world_size,
        )
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            config,
            step,
            epoch,
            student,
            ema,
            optimizer,
            initial_metrics,
            rank,
            world_size,
        )
        if rank == 0:
            LOGGER.info("Initial 2-NFE metrics: %s", json.dumps(initial_metrics, sort_keys=True))
            if run:
                run.log({f"validation/{key}": value for key, value in initial_metrics.items() if isinstance(value, (int, float))}, step=0)

    stop = False
    while step < iterations and not stop:
        train_set.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in ("LQ", "GT")}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = direct_trajectory_loss(diffusion, batch, lpips_model, config)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite direct trajectory loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), float(config["direct"]["gradient_clip"])
            )
            optimizer.step()
            ema.update(student)
            step += 1

            if step % int(config["direct"]["log_frequency"]) == 0 or step == 1:
                values = core.reduce_logs(
                    {
                        "train/total_loss": loss.detach(),
                        "train/l1": parts["l1"],
                        "train/mse": parts["mse"],
                        "train/perceptual": parts["perceptual"],
                        "train/gradient_norm": gradient_norm,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    device,
                    world_size,
                )
                if rank == 0:
                    LOGGER.info("step=%d/%d loss=%.6f grad=%.4f", step, iterations, values["train/total_loss"], values["train/gradient_norm"])
                    if run:
                        run.log(values, step=step)

            should_validate = step % validation_frequency == 0 or step == iterations
            if should_validate:
                metrics = evaluate_state(
                    config,
                    ema.state_dict_cpu(),
                    val_set,
                    device,
                    rank,
                    world_size,
                    limit=validation_limit,
                    max_edge=validation_max_edge,
                )
                improved = best_metrics is None or metrics["psnr"] > best_metrics["psnr"]
                if improved:
                    best_metrics = metrics
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        config,
                        step,
                        epoch,
                        student,
                        ema,
                        optimizer,
                        metrics,
                        rank,
                        world_size,
                    )
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    config,
                    step,
                    epoch,
                    student,
                    ema,
                    optimizer,
                    metrics,
                    rank,
                    world_size,
                )
                if rank == 0:
                    LOGGER.info("2-NFE validation step=%d metrics=%s", step, json.dumps(metrics, sort_keys=True))
                    if run:
                        run.log({f"validation/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}, step=step)
                        run.summary["best_psnr"] = best_metrics["psnr"]
                        run.summary["best_checkpoint_path"] = str((checkpoint_dir / "best.pt").resolve())
                if not smoke and step >= minimum_iterations and best_metrics["psnr"] >= target_psnr:
                    stop = True
                    break

            elif step % int(config["direct"]["checkpoint_frequency"]) == 0:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    config,
                    step,
                    epoch,
                    student,
                    ema,
                    optimizer,
                    None,
                    rank,
                    world_size,
                )
            if step >= iterations:
                break
        epoch += 1

    achieved = bool(best_metrics and best_metrics["psnr"] >= target_psnr)
    if rank == 0:
        summary = {
            "achieved": achieved,
            "target_psnr": target_psnr,
            "best_metrics": best_metrics,
            "best_checkpoint": str((checkpoint_dir / "best.pt").resolve()),
            "step": step,
        }
        core.atomic_json_save(summary, Path(config["output_dir"]) / "result.json")
        LOGGER.info("Direct training result: %s", json.dumps(summary, sort_keys=True))
        if run:
            run.summary.update(summary)
            run.finish()
    core.barrier(world_size)
    if not smoke and not achieved:
        raise RuntimeError(f"2-NFE target not reached: best_psnr={best_metrics['psnr']:.4f} < {target_psnr:.4f}")


def main():
    args = parse_args()
    config, dataset_config = load_configs(args)
    rank, local_rank, world_size, device = core.setup_distributed()
    core.configure_logging(rank, config["output_dir"])
    seed = int(config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    LOGGER.info("world_size=%d local_rank=%d commit=%s", world_size, local_rank, core.git_commit())
    try:
        run_training(config, dataset_config, args, rank, local_rank, world_size, device)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
