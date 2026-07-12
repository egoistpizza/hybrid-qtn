# `dataset/` — YAML-driven segmentation dataloader

Single generic dataset class (`GenericSegmentationDataset`) driven by a YAML config.
No per-dataset subclasses. A thin wrapper (`load_kvasir_seg`) exists for convenience.

## Output contract

`__getitem__` returns a `dict`:

```python
{
    "image": torch.FloatTensor,   # (C, H, W), float32
    "mask":  torch.FloatTensor,   # (1, H, W), float32, values in {0.0, 1.0}
    "metadata": {
        "image_path":    str,
        "mask_path":     str,
        "sample_id":     str,               # filename stem
        "original_size": tuple[int, int],   # (H0, W0) pre-transform
        "dataset_name":  str,
        "index":         int,
    },
}
```

A default `torch.utils.data.DataLoader` will collate:
- `batch["image"]` → `(B, C, H, W)`
- `batch["mask"]`  → `(B, 1, H, W)`
- `batch["metadata"]` → dict of lists (paths, ids) and stacked tensors (indices, sizes).

## YAML schema

```yaml
dataset_name:   str            # free-form label, echoed in metadata
images_dir:     str            # directory with images
masks_dir:      str            # directory with masks
image_ext:      str            # e.g. ".jpg"
mask_ext:       str            # e.g. ".jpg" or ".png"
strict_pairing: bool = true    # raise on orphan; false = warn + skip
mask_threshold: int = 127      # foreground cutoff for raw grayscale mask
transforms:                    # optional list of Albumentations specs
  - name: <ClassName>
    <kwargs>: ...
```

### Transforms

Each list entry is `{name: <AlbumentationsClass>, ...kwargs}`. Names resolve
against `albumentations.*` plus `ToTensorV2` from `albumentations.pytorch`.
Order matters. `ToTensorV2` must be last if present.

To match the standard output contract, include a `Normalize` step followed by
`ToTensorV2` (see `configs/kvasir_seg.yaml`).

## Usage

```python
from dataset import load_kvasir_seg
from torch.utils.data import DataLoader

ds = load_kvasir_seg("configs/kvasir_seg.yaml")
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

batch = next(iter(loader))
assert batch["image"].shape == (4, 3, 256, 256)
assert batch["mask"].shape  == (4, 1, 256, 256)
```

Or, with an arbitrary config dict:

```python
from dataset import GenericSegmentationDataset

ds = GenericSegmentationDataset({
    "dataset_name": "kvasir-seg",
    "images_dir":   "data/kvasir-seg/Kvasir-SEG/images",
    "masks_dir":    "data/kvasir-seg/Kvasir-SEG/masks",
    "image_ext":    ".jpg",
    "mask_ext":     ".jpg",
    "transforms": [
        {"name": "Resize", "height": 256, "width": 256},
        {"name": "Normalize"},
        {"name": "ToTensorV2"},
    ],
})
```

## Adding a new dataset

Assumes the dataset can be represented as an `images/` + `masks/` folder pair
with matching stems. No new Python needed — just a new YAML:

1. Copy `configs/kvasir_seg.yaml` to `configs/<name>.yaml`.
2. Set `dataset_name`, `images_dir`, `masks_dir`, `image_ext`, `mask_ext`.
3. Tune `mask_threshold` if masks aren't grayscale-thresholded around 127.
4. Load it: `GenericSegmentationDataset(yaml.safe_load(open(path)))`.

If the dataset has a materially different layout (split subdirs, multi-class
masks, JSON annotations), that's the point where a dedicated loader or a
pairing helper belongs — start by adding a sibling to `pairing.py`.

## Adding / removing transforms

Edit the `transforms:` list in the YAML. Any Albumentations transform is
valid; kwargs are passed through verbatim. Common tweaks:

- **Resize**: change `height` / `width`.
- **Augmentation**: add `HorizontalFlip`, `VerticalFlip`, `Rotate`,
  `RandomBrightnessContrast`, etc.
- **Normalization**: `Normalize` with `mean` / `std` lists.
- **Tensor conversion**: keep `ToTensorV2` at the end.

Removing `Normalize` gives raw 0–255 float tensors; removing `ToTensorV2`
returns NumPy arrays with `HWC` layout — the loader will fall back to a
manual conversion but you lose Albumentations' guarantee.
