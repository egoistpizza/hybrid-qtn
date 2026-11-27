import zipfile
from pathlib import Path, PurePosixPath

import pytest

from scripts import download_duts
from scripts.download_duts import destination_for, verify_layout
from scripts.fetch import extract_zip


@pytest.mark.parametrize(
    "member, expected",
    [
        ("DUTS-TR/DUTS-TR-Image/ILSVRC2012_test_00000030.jpg", "train/duts_tr/images/ILSVRC2012_test_00000030.jpg"),
        ("DUTS-TR/DUTS-TR-Mask/n03764736_2729.png", "train/duts_tr/masks/n03764736_2729.png"),
        ("DUTS-TE/DUTS-TE-Image/sun_avoanrnhaakppjpt.jpg", "test/duts_te/images/sun_avoanrnhaakppjpt.jpg"),
        ("DUTS-TE/DUTS-TE-Mask/ILSVRC2012_test_00040628.png", "test/duts_te/masks/ILSVRC2012_test_00040628.png"),
    ],
)
def test_members_map_to_the_normalized_layout(member, expected):
    assert destination_for(member) == PurePosixPath(expected)


@pytest.mark.parametrize(
    "member",
    [
        "DUTS-TR/",
        "DUTS-TR/DUTS-TR-Image/",
        "DUTS-TE/.DS_Store",
        "__MACOSX/DUTS-TE/DUTS-TE-Image/._sun_a.jpg",
        "DUTS-TR/DUTS-TE-Image/ILSVRC2012_test_00000030.jpg",
        "DUTS-TE/DUTS-TE-Edge/ILSVRC2012_test_00040628.png",
    ],
)
def test_irrelevant_members_are_skipped(member):
    assert destination_for(member) is None


def test_unexpected_file_type_is_rejected():
    with pytest.raises(ValueError, match="expected a .png file"):
        destination_for("DUTS-TE/DUTS-TE-Mask/sun_a.jpg")


def build_archive(path: Path, members: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, b"payload")
    return path


def test_extraction_and_verification_round_trip(tmp_path):
    archive = build_archive(
        tmp_path / "archive.zip",
        [
            "DUTS-TR/DUTS-TR-Image/a.jpg",
            "DUTS-TR/DUTS-TR-Mask/a.png",
            "DUTS-TE/DUTS-TE-Image/b.jpg",
            "DUTS-TE/DUTS-TE-Mask/b.png",
            "DUTS-TE/.DS_Store",
        ],
    )
    data_root = tmp_path / "duts"
    extract_zip(archive, data_root, destination_for)
    verify_layout(data_root, {"train/duts_tr": 1, "test/duts_te": 1})
    assert sorted(path.relative_to(data_root).as_posix() for path in data_root.rglob("*") if path.is_file()) == [
        "test/duts_te/images/b.jpg",
        "test/duts_te/masks/b.png",
        "train/duts_tr/images/a.jpg",
        "train/duts_tr/masks/a.png",
    ]


def test_verification_reports_unpaired_and_missing_files(tmp_path):
    archive = build_archive(
        tmp_path / "archive.zip", ["DUTS-TE/DUTS-TE-Image/a.jpg", "DUTS-TE/DUTS-TE-Mask/b.png"]
    )
    extract_zip(archive, tmp_path / "duts", destination_for)
    with pytest.raises(RuntimeError, match="unpaired") as failure:
        verify_layout(tmp_path / "duts", {"test/duts_te": 2})
    assert "expected 2 images, found 1" in str(failure.value)


def test_expected_counts_follow_the_benchmark_protocol():
    assert download_duts.EXPECTED_COUNTS == {"train/duts_tr": 10553, "test/duts_te": 5019}


def test_archives_are_pinned_by_size_and_digest():
    for archive in download_duts.ARCHIVES:
        assert archive.size_bytes > 0
        assert len(archive.sha256) == 64
