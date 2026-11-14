import pytest

from dataset import RandomSplit, SplitFractions, load_config, make_splits
from tests.real_data import CONFIG_DIR, require_on_disk


class TestSplitFractions:
    def test_default_is_seventy_fifteen_fifteen(self):
        fractions = SplitFractions()
        assert (round(fractions.train, 10), fractions.val, fractions.test) == (0.7, 0.15, 0.15)

    def test_sizes_cover_every_sample(self):
        assert SplitFractions().sizes(1000) == (700, 150, 150)
        assert sum(SplitFractions().sizes(1171)) == 1171

    @pytest.mark.parametrize("val, test", [(0.0, 0.15), (0.5, 0.5), (0.15, 1.0)])
    def test_invalid_fractions_raise(self, val, test):
        with pytest.raises(ValueError):
            SplitFractions(val=val, test=test)


class TestKvasirSplit:
    def test_kvasir_split_is_seventy_fifteen_fifteen(self):
        splits = make_splits(require_on_disk("kvasir_seg"))
        test_set = splits.test_sets["test"].dataset
        assert (len(splits.train), len(splits.val.dataset), len(test_set)) == (700, 150, 150)

    def test_kvasir_config_declares_no_group_map(self):
        assert isinstance(load_config("kvasir_seg", CONFIG_DIR).split, RandomSplit)
