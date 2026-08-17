import json
import math
import socket

import pytest

from pure_gaze_typing.protocol import (
    GazeSample,
    Heartbeat,
    UdpReceiver,
    decode_message,
    encode_message,
)


def test_gaze_sample_round_trip_rejects_nonfinite_coordinates():
    sample = GazeSample(
        10.0,
        True,
        True,
        False,
        0.9,
        29.8,
        "cal-1",
        "gaze-grid-v1",
        640.0,
        360.0,
    )
    assert decode_message(encode_message(sample)) == sample
    with pytest.raises(ValueError, match="finite"):
        GazeSample(
            10.0,
            True,
            True,
            False,
            0.9,
            30.0,
            "cal-1",
            "gaze-grid-v1",
            math.nan,
            1.0,
        )


def test_heartbeat_round_trip_and_unknown_fields_are_rejected():
    heartbeat = Heartbeat(
        4.0,
        True,
        True,
        "cal-1",
        "gaze-grid-v1",
        30.0,
        streaming=False,
    )
    assert decode_message(encode_message(heartbeat)) == heartbeat
    assert not heartbeat.streaming
    payload = json.loads(encode_message(heartbeat).decode("utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        decode_message(json.dumps(payload).encode("utf-8"))


def test_receiver_online_state_expires_after_two_seconds():
    receiver = UdpReceiver.__new__(UdpReceiver)
    receiver.last_received_at = 5.0
    assert receiver.is_online(6.9)
    assert not receiver.is_online(7.01)


def test_receiver_drains_valid_datagrams_and_warns_on_bad_packets():
    warnings: list[str] = []
    receiver = UdpReceiver(port=0, warning_callback=warnings.append)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(b"bad json", ("127.0.0.1", receiver.port))
        sample = GazeSample(
            10.0,
            False,
            False,
            False,
            0.0,
            0.0,
            "cal-1",
            "gaze-grid-v1",
        )
        sender.sendto(encode_message(sample), ("127.0.0.1", receiver.port))
        assert receiver.poll(now=12.0) == [sample]
        assert receiver.last_received_at == 12.0
        assert warnings
    finally:
        sender.close()
        receiver.close()
