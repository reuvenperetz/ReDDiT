"""Lazy paired loader for the DLL low-light enhancement dataset."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class DLLDataset(Dataset):
    """DLL pairs without caching multi-megapixel images in DDP workers."""

    def __init__(self, opt, train):
        self.root = Path(opt["root"])
        self.train = bool(train)
        self.patch_size = int(opt.get("patch_size", 96))
        self.use_crop = bool(opt.get("use_crop", self.train))
        self.use_flip = bool(opt.get("use_flip", False))
        self.use_rot = bool(opt.get("use_rot", False))
        self.seed = int(opt.get("seed", 1337))
        self.epoch = 0
        self.pairs = self._find_pairs()
        expected = opt.get("expected_pairs")
        if expected is not None and len(self.pairs) != int(expected):
            raise RuntimeError(
                f"DLL pair count mismatch under {self.root}: "
                f"expected {expected}, found {len(self.pairs)}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _find_pairs(self):
        pairs = []
        if self.train:
            for low in sorted(self.root.glob("*_LOW.jpg")):
                pairs.append((low, low.with_name(low.name.replace("_LOW.jpg", "_GT.jpg"))))
            for low in sorted(self.root.glob("*_DLL.jpg")):
                pairs.append((low, low.with_name(low.name.replace("_DLL.jpg", "_GT_FINAL.jpg"))))

            long_low = sorted(self.root.glob("CAPTURE-DLL-*.jpg")) + sorted(
                self.root.glob("CAPTURE-NLL-*.jpg")
            )
            long_gt = sorted(self.root.glob("CAPTURE-GT_FINAL-*.jpg")) + sorted(
                self.root.glob("CAPTURE-GT_INITIAL-*.jpg")
            )

            def by_timestamp(paths):
                mapping = {}
                for path in paths:
                    if "TIMESTAMP-" not in path.stem:
                        raise RuntimeError(f"Missing DLL timestamp key: {path}")
                    key = path.stem.rsplit("TIMESTAMP-", 1)[1]
                    if key in mapping:
                        raise RuntimeError(f"Duplicate DLL timestamp key {key}: {path}")
                    mapping[key] = path
                return mapping

            low_by_timestamp = by_timestamp(long_low)
            gt_by_timestamp = by_timestamp(long_gt)
            if low_by_timestamp.keys() != gt_by_timestamp.keys():
                missing_gt = sorted(low_by_timestamp.keys() - gt_by_timestamp.keys())
                missing_low = sorted(gt_by_timestamp.keys() - low_by_timestamp.keys())
                raise RuntimeError(
                    f"Long-form DLL timestamp mismatch: missing_gt={missing_gt[:3]}, "
                    f"missing_low={missing_low[:3]}"
                )
            pairs.extend(
                (low_by_timestamp[key], gt_by_timestamp[key])
                for key in sorted(low_by_timestamp)
            )
        else:
            low_root = self.root / "low"
            gt_root = self.root / "gt"
            pairs = [(low, gt_root / low.name) for low in sorted(low_root.glob("*.jpg"))]

        if not pairs:
            raise RuntimeError(f"No DLL pairs found under {self.root}")
        missing = [str(gt) for _, gt in pairs if not gt.is_file()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} DLL ground-truth files; first: {missing[0]}")
        if len({str(low) for low, _ in pairs}) != len(pairs):
            raise RuntimeError("Duplicate DLL low-light paths detected")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        index = int(index)
        rng = np.random.default_rng(self.seed + self.epoch * len(self.pairs) + index)
        low_path, gt_path = self.pairs[index]
        low = _read_rgb(low_path)
        gt = _read_rgb(gt_path)
        if low.shape != gt.shape:
            raise RuntimeError(f"Pair shape mismatch: {low_path} {low.shape} vs {gt_path} {gt.shape}")

        if self.train and self.use_crop:
            height, width = low.shape[:2]
            if height < self.patch_size or width < self.patch_size:
                raise RuntimeError(f"Image is smaller than patch size {self.patch_size}: {low_path}")
            top = int(rng.integers(0, height - self.patch_size + 1))
            left = int(rng.integers(0, width - self.patch_size + 1))
            crop = np.s_[top : top + self.patch_size, left : left + self.patch_size, :]
            low, gt = low[crop], gt[crop]

        if self.train and self.use_flip and rng.random() < 0.5:
            low, gt = np.flip(low, axis=1).copy(), np.flip(gt, axis=1).copy()
        if self.train and self.use_rot:
            rotations = int(rng.integers(0, 4))
            if rotations:
                low = np.rot90(low, rotations, axes=(0, 1)).copy()
                gt = np.rot90(gt, rotations, axes=(0, 1)).copy()

        low_tensor = torch.from_numpy(np.ascontiguousarray(low.transpose(2, 0, 1))).float().div_(127.5).sub_(1.0)
        gt_tensor = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1))).float().div_(127.5).sub_(1.0)
        return {
            "LQ": low_tensor,
            "GT": gt_tensor,
            "LQ_path": str(low_path),
            "GT_path": str(gt_path),
            "index": index,
        }
