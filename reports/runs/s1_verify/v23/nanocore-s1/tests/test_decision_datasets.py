"""Dataset normalisation — every label shape, with a fake split.

Each case here is a bug that cost a GPU session when this logic lived in a
notebook cell, so each gets a test.
"""
import numpy as np
import pytest

from src.decision.datasets import (class_halves, class_quota, label_field,
                                   label_space, texts_of)


class FakeFeature:
    def __init__(self, names):
        self.names = list(names)
        self.num_classes = len(self.names)


FakeFeature.__name__ = "ClassLabel"


class Plain:
    pass


Plain.__name__ = "Value"


class FakeSplit:
    def __init__(self, columns, features=None):
        self._columns = columns
        self.features = features or {k: Plain() for k in columns}

    def __getitem__(self, key):
        return self._columns[key]


def _class_label_split():
    return FakeSplit({"text": ["a", "b", "c"], "label": [1, 0, 1]},
                     {"text": Plain(), "label": FakeFeature(["neg", "pos"])})


def test_class_label_names_come_from_the_feature():
    split = _class_label_split()
    assert label_field(split.features) == "label"
    space = label_space(split, "label")
    assert list(space.names) == ["neg", "pos"]
    assert space.ids(split, "label").tolist() == [1, 0, 1]


def test_coarse_label_wins_when_a_dataset_has_two():
    features = {"label-coarse": FakeFeature(["a", "b", "c"]),
                "label-fine": FakeFeature([str(i) for i in range(50)])}
    assert label_field(features) == "label-coarse"


def test_numeric_labels_with_a_name_column_are_compacted():
    """mteb/banking77: label is an int, label_text the name — and the ids are
    compacted to name order, not passed through."""
    split = FakeSplit({"label": [11, 3, 11], "label_text": ["card", "atm", "card"]},
                      {"label": Plain(), "label_text": Plain()})
    space = label_space(split, "label", names_field="label_text")
    assert list(space.names) == ["atm", "card"]
    assert space.ids(split, "label").tolist() == [1, 0, 1]


def test_sparse_numeric_labels_compact_to_a_dense_space():
    split = FakeSplit({"label": [70, 3, 70]}, {"label": Plain()})
    space = label_space(split, "label")
    assert list(space.names) == ["3", "70"]
    assert space.ids(split, "label").tolist() == [1, 0, 1]


def test_string_labels_are_mapped_on_their_raw_text():
    """mteb/amazon_massive_intent: label IS the name. Mapping must use the raw
    string — prettifying it first is how the first version failed its own test."""
    split = FakeSplit({"label": ["alarm_set", "audio_volume_mute", "alarm_set"]},
                      {"label": Plain()})
    space = label_space(split, "label")
    assert list(space.names) == ["alarm_set", "audio_volume_mute"]
    assert space.ids(split, "label").tolist() == [0, 1, 0]
    assert space.prompt_names() == ["alarm set", "audio volume mute"]


def test_a_test_split_reuses_the_training_label_space():
    train = FakeSplit({"label": ["b", "a", "b"]}, {"label": Plain()})
    space = label_space(train, "label")
    assert list(space.names) == ["a", "b"]
    assert space.ids(train, "label").tolist() == [1, 0, 1]

    test = FakeSplit({"label": ["b", "a"]}, {"label": Plain()})
    assert space.ids(test, "label").tolist() == [1, 0]


def test_labels_outside_the_space_fail_loudly():
    space = label_space(FakeSplit({"label": ["a", "b"]}, {"label": Plain()}), "label")
    stray = FakeSplit({"label": ["a", "mystery"]}, {"label": Plain()})
    with pytest.raises(ValueError):
        space.ids(stray, "label")
    # an override on numeric labels has no way to say which id is which name,
    # so it is refused rather than guessed
    overflow = FakeSplit({"label": [0, 77]}, {"label": Plain()})
    with pytest.raises(ValueError):
        label_space(overflow, "label", names=["c0", "c1"])


def test_label_field_fails_loudly_when_there_is_no_label():
    with pytest.raises(ValueError):
        label_field({"text": Plain()})


def test_class_quota_keeps_every_class():
    """A head slice gives one class and a stride drops the short ones."""
    labels = [0] * 100 + [1] * 60 + [2] * 4 + [3] * 2
    split = FakeSplit({"label": labels}, {"label": Plain()})
    picked = class_quota(split, "label", cap=40)
    assert sorted(set(np.asarray(labels)[picked].tolist())) == [0, 1, 2, 3]


def test_class_quota_prefers_at_least_three_rows_per_class():
    labels = list(range(50)) * 4                     # 50 classes, 4 rows each
    split = FakeSplit({"label": labels}, {"label": Plain()})
    picked = class_quota(split, "label", cap=10)     # cap smaller than classes
    assert len(set(np.asarray(labels)[picked].tolist())) == 50


def test_class_halves_give_both_splits_every_class():
    """The v16 shadow failure: splitting a concatenated per-class index at the
    midpoint partitions by class, so the two halves have disjoint label spaces.
    """
    labels = [0] * 60 + [1] * 60 + [2] * 10
    split = FakeSplit({"label": labels}, {"label": Plain()})
    first, second = class_halves(split, "label", cap=120)
    for half in (first, second):
        assert sorted(set(np.asarray(labels)[half].tolist())) == [0, 1, 2]


def test_class_halves_duplicate_a_singleton_into_both():
    """A class with one row cannot be split; dropping it from a half silently
    gives that retrain a smaller label space."""
    labels = [0] * 30 + [1] * 30 + [2]
    split = FakeSplit({"label": labels}, {"label": Plain()})
    first, second = class_halves(split, "label", cap=120)
    assert 60 in first and 60 in second


def test_class_halves_cover_the_data_once():
    labels = [0] * 30 + [1] * 30
    split = FakeSplit({"label": labels}, {"label": Plain()})
    first, second = class_halves(split, "label", cap=40)
    assert not set(first) & set(second)
    assert len(first) + len(second) <= 40


def test_texts_of_joins_a_premise_hypothesis_pair():
    split = FakeSplit({"premise": ["a cat sits"], "hypothesis": ["an animal is here"],
                       "label": [0]}, {"premise": Plain(), "hypothesis": Plain()})
    assert texts_of(split, "premise", "hypothesis") == ["a cat sits an animal is here"]
    assert texts_of(split, "premise") == ["a cat sits"]
