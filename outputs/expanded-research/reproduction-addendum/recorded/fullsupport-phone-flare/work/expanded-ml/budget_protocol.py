"""Measured training-budget accounting without model or stopping-policy imports.

The caller measures completed update durations, excludes validation time, and
decides what to execute. A minimum-update requirement never changes the cap.
"""
import math
from numbers import Integral, Real


def _positive_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class BudgetTracker:
    """Track measured updates and allow validation only at a completed epoch.

    ``record_update(seconds, samples)`` records actual work, including any final
    update that crossed the cap. It never extends the cap or invents updates.
    ``completed_epoch`` must describe all updates since the preceding completion.
    ``validation_due`` is an elapsed-training-time notification, not an estimate.
    """

    def __init__(self, train_seconds_cap, validation_interval_seconds, min_updates):
        self.train_seconds_cap = _positive_seconds(train_seconds_cap, "train_seconds_cap")
        self.validation_interval_seconds = _positive_seconds(validation_interval_seconds, "validation_interval_seconds")
        self.min_updates = _integer(min_updates, "min_updates")
        self.total_updates = 0
        self.total_samples = 0
        self.train_seconds = 0.
        self._partial_updates = 0
        self._partial_samples = 0
        self._partial_seconds = 0.
        self._epochs = []
        self._at_completed_epoch = False
        self._last_validation_train_seconds = 0.
        self._validation_count = 0

    def record_update(self, seconds, samples):
        """Record one completed optimizer update and its actual sample count."""
        seconds = _positive_seconds(seconds, "seconds")
        samples = _integer(samples, "samples", 1)
        updated_seconds = self.train_seconds + seconds
        if not math.isfinite(updated_seconds):
            raise ValueError("Accumulated training seconds must remain finite")
        self.total_updates += 1
        self.total_samples += samples
        self.train_seconds = updated_seconds
        self._partial_updates += 1
        self._partial_samples += samples
        self._partial_seconds += seconds
        self._at_completed_epoch = False

    @property
    def budget_reached(self):
        return self.train_seconds >= self.train_seconds_cap

    @property
    def min_updates_satisfied(self):
        return self.total_updates >= self.min_updates

    @property
    def remaining_train_seconds(self):
        return max(0., self.train_seconds_cap - self.train_seconds)

    @property
    def validation_due(self):
        return self._at_completed_epoch and (
            self.train_seconds - self._last_validation_train_seconds
            >= self.validation_interval_seconds
        )

    def completed_epoch(self, epoch, updates_in_epoch, samples_in_epoch):
        """Record a full epoch; no partial-epoch work is silently discarded.

        Epoch numbers may start at zero or one and must strictly increase.
        Counts must match the updates actually recorded since the last epoch.
        """
        epoch = _integer(epoch, "epoch")
        updates = _integer(updates_in_epoch, "updates_in_epoch", 1)
        samples = _integer(samples_in_epoch, "samples_in_epoch", 1)
        if self._epochs and epoch <= self._epochs[-1]["epoch"]:
            raise ValueError("Completed epoch numbers must strictly increase")
        if (updates, samples) != (self._partial_updates, self._partial_samples):
            raise ValueError("Epoch counts do not match the recorded unfinished work")
        self._epochs.append({"epoch": epoch, "updates": updates, "samples": samples,
                             "train_seconds": self._partial_seconds,
                             "cumulative_train_seconds": self.train_seconds})
        self._partial_updates = 0
        self._partial_samples = 0
        self._partial_seconds = 0.
        self._at_completed_epoch = True

    def mark_validation(self):
        """Record validation at a completed epoch, resetting the time interval.

        A caller may also validate at an epoch before the interval has elapsed;
        the helper does not prescribe model-selection or final-validation policy.
        """
        if not self._at_completed_epoch:
            raise ValueError("Validation can only be marked at a completed epoch")
        self._last_validation_train_seconds = self.train_seconds
        self._validation_count += 1

    def partial_epoch_summary(self):
        return {"present": self._partial_updates > 0, "completed": False,
                "updates": self._partial_updates, "samples": self._partial_samples,
                "train_seconds": self._partial_seconds}

    def summary(self):
        """Return JSON-ready accounting, keeping cap and eligibility separate.

        No arbitrary definition of 'near the cap' is applied. The remaining time
        and any measured overshoot are explicit, so the caller can report them.
        """
        cap_state = "cap_reached" if self.budget_reached else "cap_not_reached"
        minimum_state = "minimum_met" if self.min_updates_satisfied else "minimum_unmet"
        return {"train_seconds_cap": self.train_seconds_cap,
                "validation_interval_seconds": self.validation_interval_seconds,
                "min_updates": self.min_updates, "total_updates": self.total_updates,
                "total_samples": self.total_samples, "train_seconds": self.train_seconds,
                "budget_reached": self.budget_reached,
                "min_updates_satisfied": self.min_updates_satisfied,
                "remaining_train_seconds": self.remaining_train_seconds,
                "budget_overshoot_seconds": max(0., self.train_seconds - self.train_seconds_cap),
                "termination_eligibility": {
                    "state": f"{cap_state}_{minimum_state}",
                    "budget_stop_required": self.budget_reached,
                    "minimum_update_requirement_met": self.min_updates_satisfied},
                "completed_epochs": [dict(epoch) for epoch in self._epochs],
                "partial_epoch": self.partial_epoch_summary(),
                "validation_due": self.validation_due,
                "validation_count": self._validation_count,
                "last_validation_train_seconds": self._last_validation_train_seconds}


class CompletedEpochSelector:
    """Select a completed epoch using its validation score only.

    No test score or label enters this interface. Tied validation scores preserve
    the previously selected epoch. Both properties are None before a candidate.
    """

    def __init__(self):
        self._best_epoch = None
        self._best_score = None

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_score(self):
        return self._best_score

    def consider(self, epoch, validation_normalized_mse, *, completed: bool):
        if not isinstance(completed, bool):
            raise TypeError("completed must be a bool")
        if not completed:
            raise ValueError("Partial epochs cannot be checkpoint-selection candidates")
        epoch = _integer(epoch, "epoch", 1)
        score = validation_normalized_mse
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("validation_normalized_mse must be a finite nonnegative number")
        score = float(score)
        if not math.isfinite(score) or score < 0:
            raise ValueError("validation_normalized_mse must be a finite nonnegative number")
        if self._best_score is None or score < self._best_score:
            self._best_epoch, self._best_score = epoch, score
            return True
        return False


def self_check():
    tracker = BudgetTracker(10., 3., 4)
    tracker.record_update(2., 5)
    tracker.record_update(2., 2)
    assert not tracker.validation_due and not tracker.budget_reached
    assert tracker.partial_epoch_summary()["updates"] == 2
    assert tracker.partial_epoch_summary()["samples"] == 7
    tracker.completed_epoch(1, 2, 7)
    assert tracker.validation_due and not tracker.partial_epoch_summary()["present"]
    tracker.mark_validation()
    assert not tracker.validation_due
    tracker.record_update(1., 3)
    tracker.completed_epoch(2, 1, 3)
    assert not tracker.validation_due
    tracker.record_update(3., 4)
    assert not tracker.validation_due
    tracker.completed_epoch(3, 1, 4)
    assert tracker.validation_due and tracker.min_updates_satisfied
    tracker.mark_validation()
    tracker.record_update(2.5, 1)
    report = tracker.summary()
    assert report["budget_reached"] and report["budget_overshoot_seconds"] == .5
    assert not report["validation_due"] and report["partial_epoch"]["updates"] == 1
    assert report["partial_epoch"]["samples"] == 1 and len(report["completed_epochs"]) == 3
    insufficient = BudgetTracker(1., .5, 100)
    insufficient.record_update(1.1, 2)
    report = insufficient.summary()
    assert report["budget_reached"] and not report["min_updates_satisfied"]
    assert report["remaining_train_seconds"] == 0. and report["train_seconds_cap"] == 1.
    assert report["termination_eligibility"]["state"] == "cap_reached_minimum_unmet"
    assert report["termination_eligibility"]["budget_stop_required"]
    for seconds, samples in ((0., 1), (float("nan"), 1), (float("inf"), 1), (1., 0), (1., True)):
        before = insufficient.summary()
        try:
            insufficient.record_update(seconds, samples)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid update was accepted")
        assert insufficient.summary() == before
    try:
        insufficient.mark_validation()
    except ValueError:
        pass
    else:
        raise AssertionError("Partial-epoch validation was accepted")
    selector = CompletedEpochSelector()
    assert selector.best_epoch is None and selector.best_score is None
    for epoch, score, completed in ((1, .1, False), (0, .1, True), (1, float("nan"), True),
                                    (1, float("inf"), True), (1, -.1, True)):
        try:
            selector.consider(epoch, score, completed=completed)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid or incomplete epoch became a candidate")
        assert selector.best_epoch is None and selector.best_score is None
    assert selector.consider(1, .5, completed=True)
    assert not selector.consider(2, .5, completed=True)
    assert selector.best_epoch == 1 and selector.best_score == .5
    assert selector.consider(3, 0., completed=True)
    assert selector.best_epoch == 3 and selector.best_score == 0.
    # Opposite test rankings are kept outside the selector and cannot change it.
    for independent_test_values in ((0., 100.), (100., 0.)):
        toy = CompletedEpochSelector()
        for epoch, validation_score in enumerate((.8, .3), start=1):
            toy.consider(epoch, validation_score, completed=True)
        assert toy.best_epoch == 2 and toy.best_score == .3
        assert len(independent_test_values) == 2
    changed_validation = CompletedEpochSelector()
    for epoch, validation_score in enumerate((.1, .3), start=1):
        changed_validation.consider(epoch, validation_score, completed=True)
    assert changed_validation.best_epoch == 1
    print("BudgetTracker and CompletedEpochSelector stdlib self-check passed")


if __name__ == "__main__":
    self_check()
