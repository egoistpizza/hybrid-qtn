import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import discover_dataset_configs, load_dataset


_ROOT = Path(__file__).resolve().parent.parent

# Dataset name -> images_dir to probe before running (skip if not downloaded).
DATASET_IMAGE_DIRS = {
    "kvasir_seg": "data/kvasir-seg/Kvasir-SEG/images",
    "cvc_clinicdb": "data/cvc-clinicdb/CVC-ClinicDB/Original",
}


@pytest.mark.parametrize("dataset_name", sorted(DATASET_IMAGE_DIRS))
def test_augmentations_and_visualization(dataset_name):
    image_size = 512
    num_samples = 3

    configs = discover_dataset_configs(_ROOT / "configs")
    if dataset_name not in configs:
        pytest.skip(f"no config for {dataset_name}")
    if not (_ROOT / DATASET_IMAGE_DIRS[dataset_name]).is_dir():
        pytest.skip(f"{dataset_name} not on disk")

    config_path = configs[dataset_name]
    output_path = _ROOT / f"augmentation_preview_{dataset_name}.png"

    dataset = load_dataset(config_path)

    assert len(dataset) > 0, "Dataset must not be empty."

    sample = dataset[0]
    assert "image" in sample and "mask" in sample, "Sample dict must contain 'image' and 'mask' keys."
    assert sample["image"].shape == (3, image_size, image_size), f"Expected shape (3, {image_size}, {image_size}), got {sample['image'].shape}."
    assert sample["mask"].ndim in (2, 3), "Mask must have exactly 2 or 3 dimensions."

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples), squeeze=False)

    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for index in range(num_samples):
        sample = dataset[index]
        image_tensor = sample["image"]
        mask_tensor = sample["mask"]

        denormalized_image = torch.clamp(image_tensor * std_tensor + mean_tensor, 0, 1)
        image_array = denormalized_image.permute(1, 2, 0).numpy()
        mask_array = mask_tensor.squeeze(0).numpy() if mask_tensor.ndim == 3 else mask_tensor.numpy()

        axes[index, 0].imshow(image_array)
        axes[index, 0].set_title(f"{dataset_name} — Augmented Image {index + 1} ({image_size}x{image_size})")
        axes[index, 0].axis("off")

        axes[index, 1].imshow(mask_array, cmap="gray")
        axes[index, 1].set_title(f"Corresponding Mask {index + 1}")
        axes[index, 1].axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close(fig)

    assert output_path.exists(), f"Output image was not saved to {output_path}."
