import pytest

from dataset import pair_by_stem


class TestPairing:
    def test_pairs_matched_stems_in_sorted_order(self, fake_dataset_dir):
        pairs = pair_by_stem(
            fake_dataset_dir["images_dir"],
            fake_dataset_dir["masks_dir"],
            image_ext=".jpg",
            mask_ext=".png",
        )
        assert len(pairs) == 4
        assert [stem for _, _, stem in pairs] == sorted(fake_dataset_dir["stems"])

    def test_extension_normalization_accepts_no_leading_dot(self, fake_dataset_dir):
        pairs = pair_by_stem(
            fake_dataset_dir["images_dir"],
            fake_dataset_dir["masks_dir"],
            image_ext="jpg",
            mask_ext="png",
        )
        assert len(pairs) == 4

    def test_orphan_strict_raises(self, fake_dataset_dir):
        (fake_dataset_dir["masks_dir"] / "sample_01.png").unlink()
        with pytest.raises(FileNotFoundError):
            pair_by_stem(
                fake_dataset_dir["images_dir"],
                fake_dataset_dir["masks_dir"],
                image_ext=".jpg",
                mask_ext=".png",
                strict_pairing=True,
            )

    def test_orphan_non_strict_warns_and_skips(self, fake_dataset_dir):
        (fake_dataset_dir["masks_dir"] / "sample_01.png").unlink()
        with pytest.warns(UserWarning, match="Skipping unpaired"):
            pairs = pair_by_stem(
                fake_dataset_dir["images_dir"],
                fake_dataset_dir["masks_dir"],
                image_ext=".jpg",
                mask_ext=".png",
                strict_pairing=False,
            )
        assert len(pairs) == 3
        assert all(stem != "sample_01" for _, _, stem in pairs)

    def test_missing_images_dir_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            pair_by_stem(tmp_path / "nope", tmp_path, ".jpg", ".png")
