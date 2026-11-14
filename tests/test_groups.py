import csv

import pytest

from dataset import (
    GroupMap,
    grouped_kfold_indices,
    grouped_partition_indices,
    groups_for_samples,
    load_config,
    load_group_map,
    make_splits,
)
from tests.real_data import CONFIG_DIR, REPO_ROOT, require_on_disk


def cvc_partitions():
    splits = make_splits(require_on_disk("cvc_clinicdb"))
    return splits.train, splits.val.dataset, splits.test_sets["test"].dataset


CVC_GROUP_MAP = GroupMap(REPO_ROOT / "configs/cvc_clinicdb_sequences.csv", "sample_id", "sequence_id")


def _sequences(subset, mapping: dict[str, str]) -> set[str]:
    return {mapping[subset.dataset.samples[i][2]] for i in subset.indices}


class TestSequenceGrouping:
    def test_group_map_covers_every_cvc_frame(self):
        require_on_disk("cvc_clinicdb")
        mapping = load_group_map(CVC_GROUP_MAP)
        assert len(mapping) == 612
        assert len(set(mapping.values())) == 29

    def test_cvc_partitions_share_no_sequence(self):
        partitions = cvc_partitions()
        mapping = load_group_map(CVC_GROUP_MAP)
        train, val, test = (_sequences(s, mapping) for s in partitions)
        assert not train & val and not train & test and not val & test
        assert len(val) >= 3 and len(test) >= 3

    def test_cvc_split_covers_dataset_exactly_once(self):
        train, val, test = cvc_partitions()
        assert sorted([*train.indices, *val.indices, *test.indices]) == list(range(612))

    def test_cvc_holdout_fractions_are_near_target(self):
        _, val, test = cvc_partitions()
        assert 0.10 <= len(val) / 612 <= 0.20
        assert 0.10 <= len(test) / 612 <= 0.20

    def test_committed_map_matches_mirror_metadata(self):
        metadata = REPO_ROOT / "data/cvc-clinicdb/metadata.csv"
        if not metadata.is_file():
            pytest.skip("mirror metadata.csv not present")
        expected = {
            str(int(r["frame_id"])): str(int(r["sequence_id"]))
            for r in csv.DictReader(metadata.open(newline=""))
        }
        assert load_group_map(CVC_GROUP_MAP) == expected

    def test_ungrouped_sample_raises_rather_than_leaking(self, tmp_path):
        path = tmp_path / "partial.csv"
        path.write_text("sample_id,sequence_id\n1,1\n")
        with pytest.raises(KeyError, match="no group"):
            groups_for_samples(["1", "2"], GroupMap(path, "sample_id", "sequence_id"))

    def test_missing_group_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="group_map csv not found"):
            load_group_map(GroupMap(tmp_path / "nope.csv"))

    def test_grouped_partition_is_group_pure_on_synthetic_data(self):
        groups = [f"s{i // 10}" for i in range(100)]
        train, val, test = grouped_partition_indices(groups, (0.15, 0.15), seed=42)
        owners = [{groups[i] for i in part} for part in (train, val, test)]
        assert not owners[0] & owners[1] and not owners[0] & owners[2] and not owners[1] & owners[2]
        assert sorted([*train, *val, *test]) == list(range(100))
        assert val and test and train

    def test_grouped_partition_keeps_training_non_empty_with_few_groups(self):
        groups = ["a", "b", "c"]
        train, val, test = grouped_partition_indices(groups, (0.45, 0.45), seed=0)
        assert len(train) == len(val) == len(test) == 1

    def test_grouped_partition_rejects_too_few_groups(self):
        with pytest.raises(ValueError, match="at least 3 groups"):
            grouped_partition_indices(["a", "a", "b"], (0.15, 0.15), seed=0)

    def test_cvc_config_declares_its_sequence_map(self):
        assert load_config("cvc_clinicdb", CONFIG_DIR).split.group_map == CVC_GROUP_MAP


def synthetic_groups() -> list[str]:
    sizes = [12, 3, 25, 7, 18, 9, 30, 4, 15, 11, 6, 20, 8, 14, 5, 22]
    return [f"s{group}" for group, size in enumerate(sizes) for _ in range(size)]


def owners(groups: list[str], indices: list[int]) -> set[str]:
    return {groups[index] for index in indices}


class TestGroupedKFold:
    def test_every_group_is_tested_in_exactly_one_fold(self):
        groups = synthetic_groups()
        tested = [index for fold in range(5) for index in grouped_kfold_indices(groups, 5, fold, 0.15, seed=42)[2]]
        assert sorted(tested) == list(range(len(groups)))

    @pytest.mark.parametrize("fold", range(5))
    def test_partitions_of_a_fold_share_no_group_and_cover_every_sample(self, fold):
        groups = synthetic_groups()
        train, val, test = grouped_kfold_indices(groups, 5, fold, 0.15, seed=42)
        train_groups, val_groups, test_groups = (owners(groups, part) for part in (train, val, test))
        assert not train_groups & val_groups and not train_groups & test_groups and not val_groups & test_groups
        assert sorted([*train, *val, *test]) == list(range(len(groups)))
        assert train and val and test

    def test_folds_are_balanced_within_one_group(self):
        groups = synthetic_groups()
        sizes = [len(grouped_kfold_indices(groups, 5, fold, 0.15, seed=42)[2]) for fold in range(5)]
        assert max(sizes) - min(sizes) <= max(groups.count(group) for group in set(groups))

    def test_folds_are_deterministic_for_a_seed(self):
        groups = synthetic_groups()
        assert grouped_kfold_indices(groups, 5, 2, 0.15, seed=7) == grouped_kfold_indices(groups, 5, 2, 0.15, seed=7)

    @pytest.mark.parametrize(
        "folds, fold, message", [(1, 0, "at least 2 folds"), (5, 5, "fold must lie"), (5, -1, "fold must lie")]
    )
    def test_invalid_fold_settings_raise(self, folds, fold, message):
        with pytest.raises(ValueError, match=message):
            grouped_kfold_indices(synthetic_groups(), folds, fold, 0.15, seed=0)

    def test_too_few_groups_raise(self):
        with pytest.raises(ValueError, match="at least 7 groups"):
            grouped_kfold_indices(["a", "b", "c", "d", "e", "f"], 5, 0, 0.15, seed=0)

    def test_cvc_folds_test_every_frame_once_without_sharing_sequences(self):
        config = require_on_disk("cvc_clinicdb")
        mapping = load_group_map(CVC_GROUP_MAP)
        tested = []
        for fold in range(5):
            splits = make_splits(config, fold=fold)
            partitions = splits.train, splits.val.dataset, splits.test_sets["test"].dataset
            train, val, test = (_sequences(subset, mapping) for subset in partitions)
            assert not train & val and not train & test and not val & test
            tested += partitions[2].indices
        assert sorted(tested) == list(range(612))

    def test_fold_is_rejected_for_a_config_without_cross_validation(self):
        with pytest.raises(ValueError, match="cross_validation_folds"):
            make_splits(load_config("kvasir_seg", CONFIG_DIR), fold=0)

    def test_cvc_config_declares_five_folds(self):
        assert load_config("cvc_clinicdb", CONFIG_DIR).split.folds == 5
