"""Publish the local Kvasir-SEG and CVC-ClinicDB trees as Hugging Face datasets.

Each dataset becomes its own repo. The repo tree mirrors the on-disk layout the
download scripts already produce, so ``snapshot_download`` alone gives a working
dataset and nothing under ``configs/`` has to change:

    <ns>/kvasir-seg     Kvasir-SEG/images/*.jpg, Kvasir-SEG/masks/*.jpg,
                        Kvasir-SEG/kavsir_bboxes.json
    <ns>/cvc-clinicdb   CVC-ClinicDB/Original/*.png,
                        CVC-ClinicDB/Ground Truth/*.png,
                        class_dict.csv, metadata.csv
                        TIF/Original/*.tif, TIF/Ground Truth/*.tif  (--include-tif)

Repos are created **private** unless you pass --public. Both datasets are
third-party medical data redistributed under their own terms — check that
redistribution is permitted before making a mirror public, and keep the
citations in the generated dataset card.

Usage:
    python scripts/upload_datasets.py --dry-run
    python scripts/upload_datasets.py --dataset kvasir-seg
    python scripts/upload_datasets.py --dataset cvc-clinicdb --include-tif
    python scripts/upload_datasets.py --public
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_hub import MIRRORS, require_hub, resolve_repo_id  # noqa: E402

IGNORE_PATTERNS = ["*.zip", "*.rar", "*.part", ".cache/**", "**/.cache/**"]

# Counts the official releases ship; a mismatch means a partial local copy.
EXPECTED = {
    "kvasir-seg": [("Kvasir-SEG/images", 1000), ("Kvasir-SEG/masks", 1000)],
    "cvc-clinicdb": [
        ("CVC-ClinicDB/Original", 612),
        ("CVC-ClinicDB/Ground Truth", 612),
    ],
}

KVASIR_CARD = """---
license: other
license_name: kvasir-seg-research-use
license_link: https://datasets.simula.no/kvasir-seg/
pretty_name: Kvasir-SEG
task_categories:
- image-segmentation
tags:
- medical
- endoscopy
- colonoscopy
- polyp
- gastrointestinal
size_categories:
- 1K<n<10K
viewer: false
---

# Kvasir-SEG (file mirror)

1000 gastrointestinal polyp images with matching pixel-level segmentation
masks, from the Kvasir-SEG release by Simula Research Laboratory.

This is a **plain file mirror**, not a `datasets`-format repo: images and masks
are stored as the original JPEGs so any path-based dataloader can consume them
directly after `snapshot_download`. The dataset viewer is disabled for that
reason.

## Layout

```text
Kvasir-SEG/
  images/*.jpg          # 1000 RGB frames, variable size
  masks/*.jpg           # 1000 grayscale masks, same stem as their image
  kavsir_bboxes.json    # bounding-box annotations from the original release
```

Masks are grayscale JPEGs; binarise with a threshold (127 works cleanly)
rather than testing for equality with 255 — JPEG compression leaves soft edges.

## Usage

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="data/kvasir-seg",
)
# -> data/kvasir-seg/Kvasir-SEG/images, .../masks
```

## Source and citation

Original dataset: https://datasets.simula.no/kvasir-seg/ — released for
research and educational use. Cite the original paper, not this mirror:

```bibtex
@inproceedings{{jha2020kvasir,
  title     = {{Kvasir-SEG: A Segmented Polyp Dataset}},
  author    = {{Jha, Debesh and Smedsrud, Pia H. and Riegler, Michael A. and
               Halvorsen, Pal and de Lange, Thomas and Johansen, Dag and
               Johansen, Havard D.}},
  booktitle = {{International Conference on Multimedia Modeling (MMM)}},
  pages     = {{451--462}},
  year      = {{2020}},
  publisher = {{Springer}}
}}
```
"""

CVC_CARD = """---
license: other
license_name: cvc-clinicdb-research-use
license_link: https://polyp.grand-challenge.org/CVCClinicDB/
pretty_name: CVC-ClinicDB
task_categories:
- image-segmentation
tags:
- medical
- endoscopy
- colonoscopy
- polyp
- gastrointestinal
size_categories:
- n<1K
viewer: false
---

# CVC-ClinicDB (file mirror)

612 frames extracted from 29 colonoscopy sequences, each with a pixel-level
polyp mask (Bernal et al., 2015). Native resolution 384x288.

This is a **plain file mirror**, not a `datasets`-format repo: images and masks
are stored as files so any path-based dataloader can consume them directly
after `snapshot_download`. The dataset viewer is disabled for that reason.

## Layout

```text
CVC-ClinicDB/
  Original/*.png          # 612 RGB frames, 384x288
  Ground Truth/*.png      # 612 binary masks, same stem as their image
class_dict.csv            # background=(0,0,0), polyp=(255,255,255)
metadata.csv              # frame_id, sequence_id, image_path, mask_path
{tif_section}```

`metadata.csv` carries the **`sequence_id`** column: the 612 frames come from
only 29 sequences, so a frame-level random split leaks near-duplicate frames
across train and validation. Group by `sequence_id` for an honest split.
Its path columns were rewritten to match this repo's layout.

## Usage

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="data/cvc-clinicdb",
    allow_patterns=["CVC-ClinicDB/**", "*.csv"],   # skip the TIF tree
)
# -> data/cvc-clinicdb/CVC-ClinicDB/Original, .../Ground Truth
```

## Source and citation

The official distribution is a `.rar` behind a JavaScript-rendered Drive link on
https://polyp.grand-challenge.org/CVCClinicDB/ ; the PNG rendering here follows
the widely used Kaggle mirror `balraj98/cvcclinicdb`. Released for research use.
Cite the original paper, not this mirror:

```bibtex
@article{{bernal2015wm,
  title     = {{WM-DOVA maps for accurate polyp highlighting in colonoscopy:
               Validation vs. saliency maps from physicians}},
  author    = {{Bernal, Jorge and Sanchez, F. Javier and
               Fernandez-Esparrach, Gloria and Gil, Debora and
               Rodriguez, Cristina and Vilarino, Fernando}},
  journal   = {{Computerized Medical Imaging and Graphics}},
  volume    = {{43}},
  pages     = {{99--111}},
  year      = {{2015}},
  publisher = {{Elsevier}}
}}
```
"""

TIF_SECTION = """TIF/
  Original/*.tif          # original TIFF release
  Ground Truth/*.tif
"""

TIF_WARNING = """
> **Note on the TIF tree.** These are the original TIFFs. They declare
> `PhotometricInterpretation=1` (grayscale) with three 8-bit samples per pixel,
> a combination Pillow refuses to decode (`unknown pixel mode`). Use the PNGs
> unless you have a reader that tolerates it.
"""


def _count(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.is_file())


def _verify_local(name: str, root: Path) -> None:
    """Fail before touching the Hub if the local copy is partial."""
    problems = []
    for rel, expected in EXPECTED[name]:
        directory = root / rel
        if not directory.is_dir():
            problems.append(f"missing directory: {directory}")
            continue
        found = _count(directory)
        if found != expected:
            problems.append(f"{directory}: {found} files, expected {expected}")
    if problems:
        raise RuntimeError(
            "Local copy does not look complete, refusing to upload:\n  "
            + "\n  ".join(problems)
            + f"\nRun scripts/download_{name.replace('-', '_')}.py first."
        )


def _build_cvc_metadata(root: Path, include_tif: bool) -> str:
    """Rewrite the Kaggle metadata.csv paths to match the mirror layout."""
    source = root / "metadata.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found — it ships with the Kaggle mirror and carries "
            "the sequence_id column. Re-download, or drop --dataset cvc-clinicdb."
        )

    out = io.StringIO()
    fields = ["frame_id", "sequence_id", "image_path", "mask_path"]
    if include_tif:
        fields += ["tif_image_path", "tif_mask_path"]
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    missing = 0
    with source.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_id = row["frame_id"]
            record = {
                "frame_id": frame_id,
                "sequence_id": row["sequence_id"],
                "image_path": f"CVC-ClinicDB/Original/{frame_id}.png",
                "mask_path": f"CVC-ClinicDB/Ground Truth/{frame_id}.png",
            }
            if include_tif:
                record["tif_image_path"] = f"TIF/Original/{frame_id}.tif"
                record["tif_mask_path"] = f"TIF/Ground Truth/{frame_id}.tif"
            if not (root / record["image_path"]).is_file():
                missing += 1
                continue
            writer.writerow(record)

    if missing:
        print(f"  WARNING dropped {missing} metadata row(s) with no image on disk")
    return out.getvalue()


def _card(name: str, repo_id: str, include_tif: bool) -> str:
    if name == "kvasir-seg":
        return KVASIR_CARD.format(repo_id=repo_id)
    card = CVC_CARD.format(
        repo_id=repo_id,
        tif_section=TIF_SECTION if include_tif else "",
    )
    return card + (TIF_WARNING if include_tif else "")


def upload_dataset(
    name: str,
    repo_id: str | None = None,
    private: bool = True,
    include_tif: bool = False,
    token: str | None = None,
    dry_run: bool = False,
    visibility_only: bool = False,
) -> str:
    hub = require_hub()
    mirror = MIRRORS[name]
    root = mirror.default_dest.expanduser().resolve()

    try:
        repo_id = resolve_repo_id(name, repo_id)
    except RuntimeError:
        # A dry run inspects local files only, so it should still work before
        # the namespace is configured — the real upload still refuses.
        if not dry_run:
            raise
        repo_id = f"<your-namespace>/{mirror.repo_name}"

    patterns = list(mirror.upload_patterns)
    if include_tif:
        patterns.extend(mirror.extra_patterns)

    print("=" * 70)
    print(f"{name}  ->  {repo_id}  ({'private' if private else 'PUBLIC'})")

    # Visibility-only touches no files, so it must not require a local copy.
    extra_files: dict[str, str] = {}
    if not visibility_only:
        print(f"  local root: {root}")
        print(f"  include:    {patterns}")
        print(f"  exclude:    {IGNORE_PATTERNS}")

        _verify_local(name, root)

        extra_files["README.md"] = _card(name, repo_id, include_tif)
        if name == "cvc-clinicdb":
            extra_files["metadata.csv"] = _build_cvc_metadata(root, include_tif)

    if dry_run:
        # Same matcher upload_folder uses, so the preview cannot drift from it.
        from huggingface_hub.utils import filter_repo_objects

        relative = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
        )
        matched = list(
            filter_repo_objects(
                relative,
                allow_patterns=patterns,
                ignore_patterns=IGNORE_PATTERNS,
            )
        )
        total = sum((root / rel).stat().st_size for rel in matched)
        print(f"  DRY RUN: {len(matched)} file(s), {total / 1e6:.1f} MB")
        for rel in matched[:3]:
            print(f"    {rel}")
        if len(matched) > 3:
            print(f"    ... and {len(matched) - 3} more")
        print(f"  DRY RUN: would generate {', '.join(extra_files)}")
        return repo_id

    api = hub.HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as exc:
        raise RuntimeError(
            "Not authenticated with the Hugging Face Hub.\n"
            "  hf auth login          (huggingface_hub >= 0.34)\n"
            "  huggingface-cli login  (older versions)\n"
            "The token needs *write* permission."
        ) from exc
    print(f"  authenticated as: {user}")

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # create_repo(exist_ok=True) returns an existing repo untouched, so on a
    # re-run the `private` argument above is silently ignored. Reconcile it.
    current = api.repo_info(repo_id, repo_type="dataset").private
    if current != private:
        was = "private" if current else "public"
        becomes = "private" if private else "PUBLIC"
        print(f"  visibility: {was} -> {becomes}")
        if not private:
            print(
                "  NOTE going public cannot be undone for anything already "
                "fetched, indexed, or mirrored."
            )
        api.update_repo_settings(repo_id, repo_type="dataset", private=private)

    if visibility_only:
        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"  visibility only, files untouched: {url}")
        return repo_id

    for path_in_repo, content in extra_files.items():
        api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {path_in_repo}",
        )
        print(f"  uploaded {path_in_repo}")

    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Add {name} images and masks",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"  done: {url}")
    return repo_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(MIRRORS),
        help="Dataset to upload; repeatable. Default: both.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Full namespace/name. Only valid with a single --dataset.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repo public. Default is private.",
    )
    parser.add_argument(
        "--include-tif",
        action="store_true",
        help="Also upload CVC-ClinicDB's 264 MB original TIF tree.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF write token. Defaults to the cached login / HF_TOKEN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without contacting the Hub.",
    )
    parser.add_argument(
        "--visibility-only",
        action="store_true",
        help="Only reconcile public/private on existing repos; upload nothing. "
        "Use with --public to flip mirrors that already exist.",
    )
    args = parser.parse_args()

    args.dataset = args.dataset or sorted(MIRRORS)
    if args.repo_id and len(args.dataset) > 1:
        parser.error("--repo-id needs exactly one --dataset")
    return args


if __name__ == "__main__":
    args = _parse_args()
    for dataset in args.dataset:
        upload_dataset(
            dataset,
            repo_id=args.repo_id,
            private=not args.public,
            include_tif=args.include_tif,
            token=args.token,
            dry_run=args.dry_run,
            visibility_only=args.visibility_only,
        )
