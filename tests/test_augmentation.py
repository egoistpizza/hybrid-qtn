import matplotlib.pyplot as plt
import pytest

from tests.real_data import require_on_disk, whole_dataset

PREVIEW_SAMPLES = 3


@pytest.mark.parametrize("dataset_name", ["cvc_clinicdb", "kvasir_seg"])
def test_training_augmentations_render_preview(dataset_name, tmp_path):
    config = require_on_disk(dataset_name)
    dataset = whole_dataset(config)
    sample = dataset[0]
    assert sample["image"].shape == (3, config.image_size, config.image_size)
    assert sample["mask"].shape == (1, config.image_size, config.image_size)

    figure, axes = plt.subplots(PREVIEW_SAMPLES, 2, figsize=(10, 4 * PREVIEW_SAMPLES), squeeze=False)
    for index in range(PREVIEW_SAMPLES):
        sample = dataset[index]
        axes[index, 0].imshow(config.pipelines.denormalization(sample["image"]))
        axes[index, 1].imshow(sample["mask"][0].numpy(), cmap="gray")
        for axis in axes[index]:
            axis.axis("off")
    output_path = tmp_path / f"augmentation_preview_{dataset_name}.png"
    figure.savefig(output_path)
    plt.close(figure)

    assert output_path.stat().st_size > 0
