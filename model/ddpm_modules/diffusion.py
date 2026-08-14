"""Diffusion schedule, DDIM sampling, and ReDDiT trajectory loss."""

import math
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import gaussian_blur


def exists(value):
    return value is not None


def default(value, factory):
    return value if exists(value) else factory() if isfunction(factory) else factory


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
    cosine_s=8e-3,
):
    if schedule == "quad":
        return np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    if schedule == "linear":
        return np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    if schedule == "warmup10":
        return _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    if schedule == "warmup50":
        return _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    if schedule == "const":
        return linear_end * np.ones(n_timestep, dtype=np.float64)
    if schedule == "jsd":
        return 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    if schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = torch.cos(timesteps / (1 + cosine_s) * math.pi / 2).pow(2)
        alphas = alphas / alphas[0]
        return (1 - alphas[1:] / alphas[:-1]).clamp(max=0.999).numpy()
    raise NotImplementedError(schedule)


def make_resampled_beta_schedule(schedule_opt, master_steps):
    """Resample the teacher cumulative-alpha curve while preserving endpoints."""
    source_steps = int(schedule_opt.get("source_n_timestep", master_steps))
    source_betas = make_beta_schedule(
        schedule_opt["schedule"],
        source_steps,
        linear_start=float(schedule_opt["linear_start"]),
        linear_end=float(schedule_opt["linear_end"]),
    )
    if source_steps == master_steps:
        return source_betas

    source_alpha_bar = np.concatenate(([1.0], np.cumprod(1.0 - source_betas)))
    source_time = np.linspace(0.0, 1.0, source_steps + 1, dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, master_steps + 1, dtype=np.float64)
    target_log_alpha_bar = np.interp(target_time, source_time, np.log(source_alpha_bar))
    target_alpha_bar = np.exp(target_log_alpha_bar)
    target_betas = 1.0 - target_alpha_bar[1:] / target_alpha_bar[:-1]
    if not np.isclose(target_alpha_bar[-1], source_alpha_bar[-1], rtol=0, atol=1e-12):
        raise RuntimeError("Resampled schedule did not preserve the teacher endpoint")
    return np.clip(target_betas, 1e-12, 0.999)


class GaussianDiffusion(nn.Module):
    """Noise-level-conditioned diffusion used by ReDDiT."""

    def __init__(
        self,
        denoise_fn,
        image_size,
        num_timesteps,
        time_scale,
        w_str=0.0,
        w_gt=0.0,
        w_snr=0.0,
        w_lpips=0.0,
        channels=3,
        loss_type="l1",
        conditional=True,
        schedule_opt=None,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.image_size = int(image_size)
        self.num_timesteps = int(num_timesteps)
        self.time_scale = int(time_scale)
        self.channels = int(channels)
        self.loss_type = loss_type
        self.conditional = bool(conditional)
        self.w_str = float(w_str)
        self.w_gt = float(w_gt)
        self.w_snr = float(w_snr)
        self.w_lpips = float(w_lpips)
        self.CD = False
        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt)

    def _set_buffer(self, name, value):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if name in self._buffers:
            self._buffers[name] = tensor.to(self._buffers[name].device)
        else:
            self.register_buffer(name, tensor)

    def set_loss(self, device):
        reduction = "sum"
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction=reduction).to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction=reduction).to(device)
        else:
            raise NotImplementedError(self.loss_type)

    def set_new_noise_schedule(self, schedule_opt, device=None):
        master_steps = self.num_timesteps * self.time_scale
        configured_master = int(schedule_opt.get("master_n_timestep", master_steps))
        if configured_master != master_steps:
            raise ValueError(
                f"Stage schedule mismatch: num_timesteps*time_scale={master_steps}, "
                f"configured master_n_timestep={configured_master}"
            )
        betas = make_resampled_beta_schedule(schedule_opt, master_steps)
        alpha_bar = np.concatenate(([1.0], np.cumprod(1.0 - betas)))
        self._set_buffer("betas", betas)
        self._set_buffer("alphas_cumprod", alpha_bar)
        self._set_buffer("sqrt_alphas_cumprod", np.sqrt(alpha_bar))
        self._set_buffer("sqrt_one_minus_alphas_cumprod", np.sqrt(np.maximum(0.0, 1.0 - alpha_bar)))
        self._set_buffer("sqrt_recip_alphas_cumprod", np.sqrt(1.0 / alpha_bar))
        self._set_buffer("sqrt_recipm1_alphas_cumprod", np.sqrt(1.0 / alpha_bar - 1.0))
        if device is not None:
            self.to(device)

    @staticmethod
    def _expand(values):
        return values.view(-1, 1, 1, 1)

    def predict_start_from_noise(self, x_t, timestep, noise):
        return self._expand(self.sqrt_recip_alphas_cumprod[timestep]) * x_t - self._expand(
            self.sqrt_recipm1_alphas_cumprod[timestep]
        ) * noise

    @staticmethod
    def predict_eps(x_t, x_0, continuous_sqrt_alpha_cumprod):
        alpha = continuous_sqrt_alpha_cumprod
        return x_t / torch.sqrt(1 - alpha**2) - torch.sqrt(1 / (1 - alpha**2) - 1) * x_0

    @staticmethod
    def predict_start(x_t, continuous_sqrt_alpha_cumprod, noise):
        alpha = continuous_sqrt_alpha_cumprod
        return x_t / alpha - torch.sqrt(1 / alpha**2 - 1) * noise

    @staticmethod
    def q_sample(x_start, continuous_sqrt_alpha_cumprod, noise):
        return continuous_sqrt_alpha_cumprod * x_start + torch.sqrt(
            torch.clamp(1 - continuous_sqrt_alpha_cumprod**2, min=0)
        ) * noise

    @staticmethod
    def snr_map(x_0):
        blur_x_0 = gaussian_blur(x_0, kernel_size=[15, 15], sigma=[3.0, 3.0])
        gray_blur = blur_x_0[:, 0:1] * 0.299 + blur_x_0[:, 1:2] * 0.587 + blur_x_0[:, 2:3] * 0.114
        gray = x_0[:, 0:1] * 0.299 + x_0[:, 1:2] * 0.587 + x_0[:, 2:3] * 0.114
        return torch.abs(gray_blur - gray)

    def _sample_alpha_interval(self, start_index, end_index, batch_size, device):
        start = self.sqrt_alphas_cumprod[start_index]
        end = self.sqrt_alphas_cumprod[end_index]
        return start + (end - start) * torch.rand(batch_size, device=device)

    def loss(self, x_in, student_denoise_fn, student_steps, noise=None, lpips_func=None):
        x_0 = x_in["GT"]
        batch_size = x_0.shape[0]
        if self.num_timesteps != int(student_steps) * 2:
            raise ValueError("ReDDiT progressive loss requires an exact 2:1 teacher/student ratio")

        teacher_t = 2 * int(torch.randint(1, int(student_steps) + 1, (1,), device=x_0.device).item())
        scale = self.time_scale
        alpha_t = self._sample_alpha_interval((teacher_t - 1) * scale, teacher_t * scale, batch_size, x_0.device)
        alpha_tm1 = self._sample_alpha_interval((teacher_t - 2) * scale, (teacher_t - 1) * scale, batch_size, x_0.device)
        if teacher_t == 2:
            alpha_tm2 = torch.ones(batch_size, device=x_0.device)
        else:
            alpha_tm2 = self._sample_alpha_interval((teacher_t - 3) * scale, (teacher_t - 2) * scale, batch_size, x_0.device)

        alpha_t = self._expand(alpha_t)
        alpha_tm1 = self._expand(alpha_tm1)
        alpha_tm2 = self._expand(alpha_tm2)
        noise = default(noise, lambda: torch.randn_like(x_0))

        with torch.no_grad():
            z_t = self.q_sample(x_0, alpha_t, noise)
            eps_t, _ = self.denoise_fn(torch.cat([x_in["LQ"], z_t], dim=1), alpha_t.flatten(1))
            x0_t = self.predict_start(z_t, alpha_t, eps_t)
            z_tm1 = self.q_sample(x0_t, alpha_tm1, eps_t)
            eps_tm1, _ = self.denoise_fn(torch.cat([x_in["LQ"], z_tm1], dim=1), alpha_tm1.flatten(1))
            x0_tm1 = self.predict_start(z_tm1, alpha_tm1, eps_tm1)
            z_tm2 = self.q_sample(x0_tm1, alpha_tm2, eps_tm1)

            frac = torch.sqrt(torch.clamp(1 - alpha_tm2**2, min=0)) / torch.sqrt(
                torch.clamp(1 - alpha_t**2, min=1e-12)
            )
            if self.w_snr:
                low = x_in["LQ"]
                illumination = torch.max(low, dim=1, keepdim=True).values + 0.1
                reflectance = low / illumination
                reflectance = reflectance - self.snr_map(reflectance)
                refined = self.q_sample(reflectance, alpha_tm2, eps_tm1)
                z_tm2 = z_tm2 + self.w_snr * (refined - z_tm2)

            denominator = alpha_tm2 - frac * alpha_t
            denominator = torch.where(
                denominator.abs() < 1e-8,
                torch.full_like(denominator, 1e-8),
                denominator,
            )
            x_target = (z_tm2 - frac * z_t) / denominator
            eps_target = self.predict_eps(z_t, x_target, alpha_t)

        eps_predicted, _ = student_denoise_fn(torch.cat([x_in["LQ"], z_t], dim=1), alpha_t.flatten(1))
        x0_predicted = self.predict_start(z_t, alpha_t, eps_predicted)
        loss_x0 = F.mse_loss(x0_predicted, x_target, reduction="none").flatten(1).mean(1)
        loss_eps = F.mse_loss(eps_predicted, eps_target, reduction="none").flatten(1).mean(1)
        distill_loss = torch.maximum(loss_x0, loss_eps).mean()

        pixel_loss = torch.zeros((), device=x_0.device)
        if self.w_gt:
            gt_x0 = F.mse_loss(x0_predicted, x_0, reduction="none").flatten(1).mean(1)
            gt_eps = F.mse_loss(eps_predicted, noise, reduction="none").flatten(1).mean(1)
            pixel_loss = torch.maximum(gt_x0, gt_eps).mean()

        perceptual_loss = torch.zeros((), device=x_0.device)
        if self.w_lpips:
            if lpips_func is None:
                raise ValueError("LPIPS model is required when w_lpips is nonzero")
            perceptual_loss = lpips_func(x_0, x0_predicted).mean()

        total = distill_loss + self.w_gt * pixel_loss + self.w_lpips * perceptual_loss
        parts = {
            "distill_loss": distill_loss.detach(),
            "pixel_loss": pixel_loss.detach(),
            "perceptual_loss": perceptual_loss.detach(),
        }
        return total, parts

    @torch.no_grad()
    def ddim(self, x_in, initial_noise=None, clip_denoised=True):
        x_t = torch.randn_like(x_in) if initial_noise is None else initial_noise.clone()
        batch_size = x_in.shape[0]
        for step in range(self.num_timesteps, 0, -1):
            t_index = step * self.time_scale
            s_index = (step - 1) * self.time_scale
            alpha_t = self.sqrt_alphas_cumprod[t_index].expand(batch_size, 1)
            eps, _ = self.denoise_fn(torch.cat([x_in, x_t], dim=1), alpha_t)
            x_0 = self.predict_start(x_t, self._expand(alpha_t[:, 0]), eps)
            if clip_denoised:
                x_0 = torch.clamp(x_0, -1, 1)
                eps = self.predict_eps(x_t, x_0, self._expand(alpha_t[:, 0]))
            alpha_s = self.sqrt_alphas_cumprod[s_index]
            sigma_s = self.sqrt_one_minus_alphas_cumprod[s_index]
            x_t = alpha_s * x_0 + sigma_s * eps
        return torch.clamp(x_t, -1, 1)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, stride=1, initial_noise=None):
        if continous or stride != 1:
            raise NotImplementedError("DLL training uses the complete deterministic DDIM trajectory")
        return self.ddim(x_in, initial_noise=initial_noise)

    def forward(self, x, student_denoise_fn=None, student_steps=None, **kwargs):
        return self.loss(x, student_denoise_fn, student_steps, **kwargs)
