"""Shared Hugging Face Hub plumbing for the dataset download/upload scripts.

Both datasets are mirrored to the Hub as *file mirrors*: the repository tree is
exactly the layout the download scripts already produce under ``data/``, so a
plain ``snapshot_download`` reconstitutes a working dataset and the YAML files
under ``configs/`` keep working untouched.

    <namespace>/kvasir-seg      Kvasir-SEG/{images,masks}/*.jpg
                                Kvasir-SEG/kavsir_bboxes.json

    <namespace>/cvc-clinicdb    CVC-ClinicDB/Original/*.png
                                CVC-ClinicDB/Ground Truth/*.png
                                class_dict.csv, metadata.csv
                                TIF/... (optional, see UPLOADING.md)

Set the namespace once by editing ``HF_NAMESPACE`` below, or per-shell via the
``HQTN_HF_ORG`` environment variable. Individual runs can override everything
with ``--repo-id <namespace>/<name>``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Hugging Face user or organisation that owns the dataset mirrors.
# Edit this once after creating the repos (see scripts/upload_datasets.py).
HF_NAMESPACE = os.environ.get("HQTN_HF_ORG") or "berkaytrhn"

_NAMESPACE_UNSET = "CHANGE_ME"


@dataclass(frozen=True)
class HFMirror:
    """How one dataset is laid out on the Hub and where it lands on disk."""

    repo_name: str
    default_dest: Path
    marker: str
    # Files fetched by default. ``None`` means the whole repo.
    allow_patterns: tuple[str, ...] | None = None
    # Extra files fetched only when explicitly asked for (--include-tif etc.).
    extra_patterns: tuple[str, ...] = ()
    # Files uploaded from the local tree; keeps archives and caches out.
    upload_patterns: tuple[str, ...] = ()


MIRRORS: dict[str, HFMirror] = {
    "kvasir-seg": HFMirror(
        repo_name="kvasir-seg",
        default_dest=Path("data/kvasir-seg"),
        marker="Kvasir-SEG",
        allow_patterns=None,
        upload_patterns=("Kvasir-SEG/**",),
    ),
    "cvc-clinicdb": HFMirror(
        repo_name="cvc-clinicdb",
        default_dest=Path("data/cvc-clinicdb"),
        marker="CVC-ClinicDB",
        # The 264 MB TIF tree is the original release format but Pillow cannot
        # decode it (photometric=1 with 3 samples/pixel), so it is opt-in.
        allow_patterns=("CVC-ClinicDB/**", "*.csv", "README.md"),
        extra_patterns=("TIF/**",),
        upload_patterns=("CVC-ClinicDB/**", "class_dict.csv"),
    ),
}


def require_hub():
    """Import ``huggingface_hub`` or explain how to get it."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed.\n"
            "  pip install 'huggingface_hub[hf_transfer]>=0.26'\n"
            "  (it is also listed in requirements.txt)"
        ) from exc
    return huggingface_hub


def resolve_repo_id(name: str, repo_id: str | None = None) -> str:
    """Full ``namespace/name`` for a mirror, with an actionable error if unset."""
    if repo_id:
        return repo_id

    mirror = MIRRORS[name]
    if HF_NAMESPACE == _NAMESPACE_UNSET:
        raise RuntimeError(
            "No Hugging Face namespace configured. Either:\n"
            f"  - edit HF_NAMESPACE in {Path(__file__).as_posix()}, or\n"
            "  - export HQTN_HF_ORG=<your-hf-username-or-org>, or\n"
            f"  - pass --repo-id <namespace>/{mirror.repo_name}"
        )
    return f"{HF_NAMESPACE}/{mirror.repo_name}"


def download_mirror(
    name: str,
    dest: Path | None = None,
    repo_id: str | None = None,
    revision: str = "main",
    token: str | None = None,
    include_extra: bool = False,
) -> Path:
    """Materialise a Hub mirror under ``dest``; returns the extracted root."""
    hub = require_hub()
    mirror = MIRRORS[name]
    dest = Path(dest) if dest is not None else mirror.default_dest
    dest = dest.expanduser().resolve()
    repo_id = resolve_repo_id(name, repo_id)

    # allow_patterns=None means "the whole repo", which already covers the extras.
    patterns: list[str] | None = None
    if mirror.allow_patterns is not None:
        patterns = list(mirror.allow_patterns)
        if include_extra:
            patterns.extend(mirror.extra_patterns)

    print(f"Downloading {repo_id}@{revision}\n           -> {dest}")
    hub.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=patterns,
        token=token,
    )
    return dest / mirror.marker


def hub_error_hint(repo_id: str) -> str:
    """Message printed when a Hub fetch fails, pointing at the alternatives."""
    return (
        f"Could not fetch {repo_id} from the Hugging Face Hub.\n"
        "  - private repo?  run `hf auth login` (or pass --token)\n"
        "  - wrong id?      check --repo-id / HQTN_HF_ORG\n"
        "  - not uploaded?  run scripts/upload_datasets.py first\n"
        "  - offline?       use --source with an original mirror instead"
    )
