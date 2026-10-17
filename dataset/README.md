# `dataset/` — YAML-driven segmentation dataloader

Single generic dataset class (`GenericSegmentationDataset`) driven by a YAML config.
No per-dataset subclasses and no per-dataset loaders. `load_config` parses and validates
a config once into a typed `DatasetConfig` (`dataset/config.py`); datasets are addressed
by the stem of their YAML under `configs/`.

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
images_dir:     str            # directory with images; relative paths resolve against the repository root
masks_dir:      str            # directory with masks
image_ext:      str            # e.g. ".jpg"
mask_ext:       str            # e.g. ".jpg" or ".png"
strict_pairing: bool = true    # raise on orphan; false = warn + skip
mask_threshold: int = 127      # foreground cutoff for raw grayscale mask
align_mask_to_image: bool = false  # resize a mask whose size differs from its image (nearest); otherwise raise
train_transforms:              # Albumentations specs for training (augmentation allowed)
eval_transforms:               # deterministic specs for validation and test (validated)
transforms:                    # optional shared fallback when a stage-specific key is absent
  - name: <ClassName>
    <kwargs>: ...
group_map:                     # optional: split by group so no group crosses partitions
  csv: str                     # sample-to-group CSV, resolved like images_dir
  sample_column: str = sample_id
  group_column: str = group_id
```

### Transforms

Each list entry is `{name: <AlbumentationsClass>, ...kwargs}`. Names resolve
against `albumentations.*` plus `ToTensorV2` from `albumentations.pytorch`.
Order matters. Both pipelines must end with `ToTensorV2`, and the last `Resize`,
`RandomCrop` or `CenterCrop` step of each fixes the input size; train and eval must
agree on it, and it must be square. The model is built at this size.

## Usage

```python
from dataset import GenericSegmentationDataset, load_config
from torch.utils.data import DataLoader

config = load_config("kvasir_seg")              # DatasetConfig; config.image_size == 512
ds = GenericSegmentationDataset(config.split.source, config.pipelines)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

batch = next(iter(loader))
assert batch["image"].shape == (4, 3, 512, 512)
assert batch["mask"].shape  == (4, 1, 512, 512)
```

Train/val/test splitting lives in `dataset/splits.py` and is shared by `train.py` and
`evaluate.py`, so an evaluation run always sees the split its training run held
out:

```python
from dataset import build_splits, discover_dataset_configs

print(sorted(discover_dataset_configs()))       # ['cod', 'cvc_clinicdb', 'kvasir_seg', 'mass_roads']

splits = build_splits(["kvasir_seg"])           # 70 / 15 / 15, fixed split seed 42
splits.train                                    # a Dataset using train_transforms
splits.val, splits.test_sets                    # EvaluationSet(dataset, foreground_maps, denormalization); eval_transforms
splits.test_sets["test"]                        # holdout splits name their single test set "test"
splits.image_size                               # 512
splits = build_splits(["kvasir_seg", "cvc_clinicdb"])  # joint; test sets with the same name are concatenated
```

The split policy follows the config: `RandomSplit` (70 / 15 / 15), `GroupedSplit` when a
`group_map` is declared, `PredefinedSplit` for official sources.

`build_loader(dataset, batch_size, shuffle, seed)` is the one loader factory for training
and evaluation: shuffling and every worker's RNGs, including Albumentations', derive from
`seed`.

## Predefined splits (official benchmarks)

A benchmark that ships official train and test folders replaces `images_dir` /
`masks_dir` with named sources:

```yaml
sources:                       # every folder pair, by name
  <source>:
    images_dir: str
    masks_dir:  str            # per-source keys override the shared ones (e.g. image_ext)
train_sources: [<source>, ...] # concatenated for training
test_sources:  [<source>, ...] # each becomes a named test set, evaluated once and logged separately
validation_fraction: float     # carved from each training source with the split seed (stratified by source)
foreground_map_metrics: bool = false  # also report S-measure, E-measure, weighted F-measure and map MAE
```

- The four keys are all required together; `group_map`, `images_dir` and `masks_dir`
  cannot be combined with them.
- Training samples whose stem also appears in a test source are excluded with a
  warning, so the official test sets never leak into training or validation.
- The official test sets are never used for model selection: validation always comes
  from the training sources.

`configs/cod.yaml` uses this schema for the camouflaged object detection benchmark
(COD10K-train + CAMO-train for training; COD10K, CAMO, CHAMELEON and NC4K for testing).

Or, from a config mapping:

```python
from dataset import parse_config

config = parse_config({
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

1. Copy an existing config to `configs/<name>.yaml`.
2. Set `dataset_name`, `images_dir`, `masks_dir`, `image_ext`, `mask_ext`.
   **Repoint all four** — a half-edited copy silently loads the source dataset.
3. Tune `mask_threshold` if masks aren't grayscale-thresholded around 127.
4. Keep `train_transforms` and `eval_transforms` identical to the other medical
   configs, or cross-dataset comparisons are confounded. `TestDatasetConsistency`
   enforces this. `eval_transforms` may only contain deterministic transforms
   (Resize, LongestMaxSize, SmallestMaxSize, PadIfNeeded, CenterCrop, Normalize,
   ToTensorV2); anything else raises when the config is parsed, as do unknown keys.

That is the whole procedure. `discover_dataset_configs()` picks the file up
automatically, so `--dataset <name>` works in `train.py` and `evaluate.py`
with no code change.

Official train/test folders need no code either: declare them as predefined
sources (see "Predefined splits" above). Multi-class masks or JSON annotations
are the point where a dedicated loader or pairing helper belongs; start by
adding a sibling to `pairing.py`.

## Adding / removing transforms

Edit the `train_transforms:` / `eval_transforms:` lists in the YAML. Augmentation belongs in `train_transforms` only. Any Albumentations transform is
valid; kwargs are passed through verbatim. Common tweaks:

- **Resize**: change `height` / `width`.
- **Augmentation**: add `HorizontalFlip`, `VerticalFlip`, `Rotate`,
  `RandomBrightnessContrast`, etc.
- **Normalization**: `Normalize` with `mean` / `std` lists.
- **Tensor conversion**: keep `ToTensorV2` at the end; a pipeline without it is rejected.

Removing `Normalize` gives raw 0–255 float tensors.
