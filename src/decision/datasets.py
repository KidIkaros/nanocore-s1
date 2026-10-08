"""Normalising public datasets into (texts, ids, names) — the shapes we meet.

This logic lived in a notebook cell and broke four times in a row: head-of-file
caps on class-ordered data (banking77: 17 of 77 classes), strided caps that
dropped rare classes (62 of 77), an id-mapping assumed present that was not, and
a label column holding a *string* in one dataset and an int in another. None of
it was covered by a test, so every failure cost a GPU session.

Four label shapes appear in the datasets we use:

- ``ClassLabel`` — names live in the feature, values are ids
- numeric with a parallel name column — ``mteb/banking77`` (``label`` int,
  ``label_text`` name)
- numeric without one — names are the ids themselves
- string — ``mteb/amazon_massive_intent``, where ``label`` *is* the name

Two things are deliberately separate. **Mapping** must use the raw label text,
because that is what the data contains; **prompting** may prettify it, because
that is what the encoder reads. Conflating them is how the first version of this
module failed its own test: it looked up ``alarm set`` in data holding
``alarm_set``.

Ids are always compacted to ``0..k-1`` in name order, whatever the raw values
are, so a dataset with sparse or string labels normalises the same way.

Everything here takes a duck-typed split (``features``, ``__getitem__``), so it
is testable without Hugging Face or a download.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def label_field(features: Dict, wanted: str = "label") -> str:
    """The label column: the named one, else the smallest ClassLabel.

    TREC-style datasets carry a coarse and a fine label; the coarse one (fewer
    classes) is the task we want.
    """
    if wanted in features:
        return wanted
    candidates = [name for name, feature in features.items()
                  if _is_class_label(feature) and feature.num_classes > 1]
    if not candidates:
        raise ValueError(f"no label column: neither {wanted!r} nor a ClassLabel")
    return min(candidates, key=lambda name: features[name].num_classes)


@dataclass(frozen=True)
class LabelSpace:
    """Names in id order, plus the raw-value → compact-id map.

    The map cannot be rebuilt from the names alone: a numeric label column
    holds arbitrary ids (``mteb/banking77``'s ``label`` is an int paired with a
    name), so the *train* split has to hand its map to the test split. Carrying
    it as an object makes that explicit — passing bare names was ambiguous
    enough to produce a string-keyed index for integer labels.
    """
    names: Tuple[str, ...]
    index: Dict

    def ids(self, split, field: str) -> np.ndarray:
        """Compact ids for a split drawn from this label space."""
        values = list(split[field])
        missing = [v for v in values if _key(v) not in self.index]
        if missing:
            raise ValueError(
                f"labels outside the {len(self.names)}-name space: "
                f"{sorted(set(map(str, missing)))[:5]}")
        return np.asarray([self.index[_key(v)] for v in values], dtype=int)

    def prompt_names(self) -> List[str]:
        """The names as the encoder should read them (``lost_card`` → ``lost card``)."""
        return [str(n).replace("_", " ") for n in self.names]


def label_space(split, field: str, names: Optional[Sequence[str]] = None,
                names_field: Optional[str] = None) -> LabelSpace:
    """Build the label space from a split, optionally overriding the names."""
    if names is not None:
        values = list(split[field])
        raw = tuple(str(n) for n in names)
        if _is_class_label(split.features[field]):
            index = {i: i for i in range(len(raw))}
        elif _is_numeric(values):
            # Numeric ids carry no names, so an override cannot say which id is
            # which name. Guessing from the observed values would map a stray id
            # onto a real label, so this fails closed instead.
            raise ValueError(
                "a names override needs a ClassLabel or a names_field to map "
                "values to names; numeric labels have neither")
        else:
            index = {str(v): i for i, v in enumerate(raw)}
        return LabelSpace(raw, index)
    raw, index = _raw_names_and_index(split, field, names_field)
    return LabelSpace(tuple(raw), index)


def class_quota(split, field: str, cap: int) -> List[int]:
    """Indices giving every class a share, up to ``cap`` rows in total.

    A head-of-file slice of a class-ordered dataset is one class; a strided
    slice loses every class shorter than the stride. A quota keeps all of them.
    """
    values = np.asarray(split[field])
    classes = np.unique(values)
    per_class = max(3, cap // len(classes))
    picked: List[int] = []
    for cls in classes:
        picked += np.flatnonzero(values == cls)[:per_class].tolist()
    return picked


def texts_of(split, text_field: str = "text",
             pair_field: Optional[str] = None) -> List[str]:
    """One text field, or a premise/hypothesis pair joined into one string."""
    texts = list(split[text_field])
    if pair_field:
        texts = [f"{a} {b}" for a, b in zip(texts, split[pair_field])]
    return texts


def _raw_names_and_index(split, field: str, names_field: Optional[str]) -> tuple:
    """Raw names in id order, plus a raw-value → compact-id map."""
    feature = split.features[field]
    values = list(split[field])
    if _is_class_label(feature):
        return list(feature.names), {i: i for i in range(len(feature.names))}
    if _is_numeric(values) and names_field:
        paired = {_key(i): text for i, text in zip(values, split[names_field])}
        ordered_ids = sorted(paired)
        return ([paired[i] for i in ordered_ids],
                {i: position for position, i in enumerate(ordered_ids)})
    if _is_numeric(values):
        ordered = sorted({int(v) for v in values})
        return [str(v) for v in ordered], {v: i for i, v in enumerate(ordered)}
    ordered = sorted({str(v) for v in values})
    return ordered, {name: i for i, name in enumerate(ordered)}


def _key(value):
    """One key space for the index: integers stay integers, everything else is text."""
    return int(value) if isinstance(value, (int, np.integer)) else str(value)


def _is_class_label(feature) -> bool:
    return feature.__class__.__name__ == "ClassLabel"


def _is_numeric(values: Sequence) -> bool:
    if not len(values):
        return False
    try:
        int(values[0])
    except (TypeError, ValueError):
        return False
    return True
