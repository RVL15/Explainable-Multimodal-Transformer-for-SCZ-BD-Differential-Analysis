import pytest
from app.models.pipeline import get_subject_level_partitions

def test_subject_split_guarantees_zero_leakage():
    # Synthetic cohort of 20 subjects
    subjects = []
    for i in range(10):
        subjects.append({"subject_id": f"SCZ_{i:02d}", "diagnostic_label": "SCZ"})
    for i in range(10):
        subjects.append({"subject_id": f"BD_{i:02d}", "diagnostic_label": "BD"})

    train_subs, val_subs, test_subs = get_subject_level_partitions(
        subjects,
        test_ratio=0.2,
        val_ratio=0.2,
        random_seed=42
    )

    train_ids = {s["subject_id"] for s in train_subs}
    val_ids = {s["subject_id"] for s in val_subs}
    test_ids = {s["subject_id"] for s in test_subs}

    # Strict partition isolation assertions
    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert len(test_ids) > 0
    assert train_ids.isdisjoint(val_ids), "Data leakage between train and val!"
    assert train_ids.isdisjoint(test_ids), "Data leakage between train and test!"
    assert val_ids.isdisjoint(test_ids), "Data leakage between val and test!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(subjects)
