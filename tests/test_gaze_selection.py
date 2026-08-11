import pytest

from pure_gaze_typing.gaze_selection import DwellSelector, TripleBlinkDetector


def test_short_gaze_excursion_preserves_progress_but_long_excursion_resets():
    selector = DwellSelector(1.0)
    selector.update(0.0, "target_0", valid=True, blink=False)
    selector.update(0.6, "target_0", valid=True, blink=False)
    assert selector.update(0.7, None, valid=True, blink=False).progress == pytest.approx(0.6)
    assert selector.update(0.85, "target_0", valid=True, blink=False).progress == pytest.approx(0.6)
    assert selector.update(1.2, None, valid=True, blink=False).progress == 0.0


def test_trigger_requires_leaving_target_before_rearming():
    selector = DwellSelector(1.0)
    selector.update(0.0, "target_0", valid=True, blink=False)
    assert selector.update(1.0, "target_0", valid=True, blink=False).triggered_target_id == "target_0"
    assert not selector.update(2.0, "target_0", valid=True, blink=False).armed
    selector.update(2.1, None, valid=True, blink=False)
    assert selector.update(2.41, None, valid=True, blink=False).armed


def test_blink_freezes_progress_and_connection_reset_clears_it():
    selector = DwellSelector(1.0)
    selector.update(0.0, "target_0", valid=True, blink=False)
    selector.update(0.5, "target_0", valid=True, blink=False)
    assert selector.update(0.7, None, valid=False, blink=True).progress == pytest.approx(0.5)
    selector.reset()
    assert selector.update(0.8, "target_0", valid=True, blink=False).progress == 0.0


def test_three_complete_blinks_within_window_trigger_once():
    detector = TripleBlinkDetector()
    events = []
    for start in (0.0, 0.6, 1.2):
        detector.update(start, face_detected=True, blink=True)
        events.append(detector.update(start + 0.12, face_detected=True, blink=False))
    assert [event.count for event in events] == [1, 2, 3]
    assert events[-1].triple_blink
    assert not detector.update(1.4, face_detected=True, blink=False).triple_blink


def test_long_closure_and_face_loss_do_not_count_as_blinks():
    detector = TripleBlinkDetector()
    detector.update(0.0, face_detected=True, blink=True)
    assert detector.update(0.8, face_detected=True, blink=False).count == 0
    detector.update(1.0, face_detected=True, blink=True)
    assert detector.update(1.1, face_detected=False, blink=False).count == 0


def test_back_target_never_uses_less_than_1_2_seconds():
    selector = DwellSelector(0.5, target_dwell_seconds={"back": 1.2})
    selector.update(0.0, "back", valid=True, blink=False)
    assert selector.update(0.6, "back", valid=True, blink=False).triggered_target_id is None
    assert selector.update(1.2, "back", valid=True, blink=False).triggered_target_id == "back"
