import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class MaskFileSamples(Dataset):
    """Fixed 16x16 samples whose ground-truth files on disk have other sizes."""

    def __init__(self, directory, sizes=((30, 40), (22, 26), (18, 18))) -> None:
        self.paths = []
        for index, size in enumerate(sizes):
            path = directory / f"sample_{index}.png"
            ground_truth = np.zeros(size, np.uint8)
            ground_truth[5:15, 6:16] = 255
            Image.fromarray(ground_truth).save(path)
            self.paths.append(str(path))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        mask = torch.zeros(1, 16, 16)
        mask[:, 4:10, 4:10] = 1.0
        return {
            "image": mask.repeat(3, 1, 1),
            "mask": mask,
            "metadata": {"mask_path": self.paths[index], "sample_id": f"sample_{index}"},
        }


def small_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 1, 1))
