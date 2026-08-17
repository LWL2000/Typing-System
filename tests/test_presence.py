from pure_gaze_typing.presence import PresenceEvent, SeatReturnDetector


def test_sustained_face_loss_then_return_requests_reentry_calibration():
    detector = SeatReturnDetector(absence_seconds=1.5)

    assert detector.update(0.0, face_detected=True, active=True) is PresenceEvent.NONE
    assert detector.update(0.5, face_detected=False, active=True) is PresenceEvent.NONE
    assert detector.update(1.9, face_detected=False, active=True) is PresenceEvent.NONE
    assert detector.update(2.0, face_detected=False, active=True) is PresenceEvent.LEFT
    assert detector.update(2.1, face_detected=False, active=True) is PresenceEvent.NONE
    assert detector.update(2.2, face_detected=True, active=True) is PresenceEvent.RETURNED
    assert detector.update(2.3, face_detected=True, active=True) is PresenceEvent.NONE


def test_brief_face_loss_and_inactive_output_do_not_request_recalibration():
    detector = SeatReturnDetector(absence_seconds=1.5)

    detector.update(0.0, face_detected=False, active=True)
    assert detector.update(1.0, face_detected=True, active=True) is PresenceEvent.NONE
    detector.update(2.0, face_detected=False, active=False)
    assert detector.update(4.0, face_detected=True, active=False) is PresenceEvent.NONE

