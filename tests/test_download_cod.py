import sys
import zipfile
from pathlib import Path, PurePosixPath

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from scripts import download_cod
from scripts.download_cod import destination_for, extract_archive, verify_layout


@pytest.mark.parametrize(
    "member, expected",
    [
        ("COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-1.jpg", "train/cod10k/images/COD10K-CAM-1-Aquatic-1-BatFish-1.jpg"),
        ("COD-TrainDataset/GT/camourflage_00012.png", "train/camo/masks/camourflage_00012.png"),
        ("COD-TestDataset/NC4K/Imgs/1.jpg", "test/nc4k/images/1.jpg"),
        ("COD-TestDataset/CHAMELEON/GT/animal-1.png", "test/chameleon/masks/animal-1.png"),
    ],
)
def test_members_map_to_the_normalized_layout(member, expected):
    assert destination_for(member) == PurePosixPath(expected)


@pytest.mark.parametrize(
    "member",
    [
        "COD-TrainDataset/",
        "COD-TrainDataset/Imgs/",
        "__MACOSX/COD-TrainDataset/Imgs/._COD10K-CAM-1.jpg",
        "COD-TestDataset/COD10K/Edge/COD10K-CAM-1.png",
        "COD-TestDataset/NC4K/Instance/1.png",
        "COD-TestDataset/UNKNOWN/Imgs/1.jpg",
    ],
)
def test_irrelevant_members_are_skipped(member):
    assert destination_for(member) is None


def test_unknown_training_prefix_is_rejected():
    with pytest.raises(ValueError, match="matches none of the prefixes"):
        destination_for("COD-TrainDataset/Imgs/other_001.jpg")


def test_unexpected_file_type_is_rejected():
    with pytest.raises(ValueError, match="expected a .png file"):
        destination_for("COD-TestDataset/CAMO/GT/camourflage_00001.jpg")


def build_archive(path: Path, members: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, b"payload")
    return path


def test_extraction_and_verification_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(download_cod, "EXPECTED_COUNTS", {"train/cod10k": 1, "test/nc4k": 1})
    archive = build_archive(
        tmp_path / "archive.zip",
        [
            "COD-TrainDataset/Imgs/COD10K-CAM-1.jpg",
            "COD-TrainDataset/GT/COD10K-CAM-1.png",
            "COD-TestDataset/NC4K/Imgs/7.jpg",
            "COD-TestDataset/NC4K/GT/7.png",
            "COD-TestDataset/NC4K/Instance/7.png",
            "__MACOSX/COD-TestDataset/NC4K/Imgs/._7.jpg",
        ],
    )
    data_root = tmp_path / "cod"
    extract_archive(archive, data_root)
    verify_layout(data_root)
    assert sorted(path.relative_to(data_root).as_posix() for path in data_root.rglob("*.*")) == [
        "test/nc4k/images/7.jpg",
        "test/nc4k/masks/7.png",
        "train/cod10k/images/COD10K-CAM-1.jpg",
        "train/cod10k/masks/COD10K-CAM-1.png",
    ]


def test_verification_reports_unpaired_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(download_cod, "EXPECTED_COUNTS", {"test/camo": 2})
    archive = build_archive(
        tmp_path / "archive.zip",
        ["COD-TestDataset/CAMO/Imgs/camourflage_00001.jpg", "COD-TestDataset/CAMO/GT/camourflage_00002.png"],
    )
    extract_archive(archive, tmp_path / "cod")
    with pytest.raises(RuntimeError, match="unpaired") as failure:
        verify_layout(tmp_path / "cod")
    assert "expected 2 images, found 1" in str(failure.value)


def test_expected_counts_follow_the_benchmark_protocol():
    assert download_cod.EXPECTED_COUNTS == {
        "train/cod10k": 3040,
        "train/camo": 1000,
        "test/cod10k": 2026,
        "test/camo": 250,
        "test/chameleon": 76,
        "test/nc4k": 4121,
    }
