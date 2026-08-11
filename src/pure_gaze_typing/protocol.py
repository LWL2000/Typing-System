from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import ipaddress
import json
import math
import socket
import time
from typing import Callable


PROTOCOL_VERSION = 1
MAX_DATAGRAM_BYTES = 65_535


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class GazeSample:
    timestamp: float
    valid: bool
    face_detected: bool
    blink: bool
    quality: float
    fps: float
    calibration_id: str
    layout_version: str
    screen_x: float | None = None
    screen_y: float | None = None
    raw_x: float | None = None
    raw_y: float | None = None

    def __post_init__(self) -> None:
        for name in ("timestamp", "quality", "fps"):
            _require_finite(name, getattr(self, name))
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        if self.fps < 0.0:
            raise ValueError("fps must be non-negative")
        coordinates = (self.screen_x, self.screen_y, self.raw_x, self.raw_y)
        for name, value in zip(("screen_x", "screen_y", "raw_x", "raw_y"), coordinates):
            if value is not None:
                _require_finite(name, value)
        if self.valid and (self.screen_x is None or self.screen_y is None):
            raise ValueError("valid gaze samples require screen coordinates")
        if not self.valid and any(value is not None for value in coordinates):
            raise ValueError("invalid gaze samples cannot contain coordinates")


@dataclass(frozen=True)
class Heartbeat:
    timestamp: float
    camera_ok: bool
    calibration_ready: bool
    calibration_id: str
    layout_version: str
    fps: float
    error: str | None = None

    def __post_init__(self) -> None:
        _require_finite("timestamp", self.timestamp)
        _require_finite("fps", self.fps)
        if self.fps < 0.0:
            raise ValueError("fps must be non-negative")


ProtocolMessage = GazeSample | Heartbeat


def encode_message(message: ProtocolMessage) -> bytes:
    if isinstance(message, GazeSample):
        message_type = "gaze"
    elif isinstance(message, Heartbeat):
        message_type = "heartbeat"
    else:
        raise TypeError("unsupported protocol message")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": message_type,
        **asdict(message),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_DATAGRAM_BYTES:
        raise ValueError("protocol datagram exceeds 64 KiB")
    return encoded


def decode_message(payload: bytes) -> ProtocolMessage:
    if len(payload) > MAX_DATAGRAM_BYTES:
        raise ValueError("protocol datagram exceeds 64 KiB")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid UTF-8 JSON datagram") from error
    if not isinstance(data, dict):
        raise ValueError("protocol message must be a JSON object")
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    message_type = data.get("message_type")
    message_class: type[GazeSample] | type[Heartbeat]
    if message_type == "gaze":
        message_class = GazeSample
    elif message_type == "heartbeat":
        message_class = Heartbeat
    else:
        raise ValueError("unsupported message type")
    expected = {field.name for field in fields(message_class)} | {
        "protocol_version",
        "message_type",
    }
    if set(data) != expected:
        raise ValueError("protocol message fields do not match schema")
    values = {key: value for key, value in data.items() if key not in {"protocol_version", "message_type"}}
    try:
        return message_class(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {message_type} message: {error}") from error


def _validate_loopback(host: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("UDP host must be a numeric loopback address") from error
    if not address.is_loopback:
        raise ValueError("UDP transport is restricted to loopback")


class UdpPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 9101) -> None:
        _validate_loopback(host)
        self._address = (host, int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message: ProtocolMessage) -> None:
        self._socket.sendto(encode_message(message), self._address)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "UdpPublisher":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class UdpReceiver:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9101,
        *,
        warning_callback: Callable[[str], None] | None = None,
    ) -> None:
        _validate_loopback(host)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, int(port)))
        self._socket.setblocking(False)
        self.port = int(self._socket.getsockname()[1])
        self.last_received_at: float | None = None
        self._warning_callback = warning_callback or (lambda _message: None)

    def poll(self, *, now: float | None = None) -> list[ProtocolMessage]:
        received_at = time.monotonic() if now is None else float(now)
        messages: list[ProtocolMessage] = []
        while True:
            try:
                payload, _address = self._socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
            except BlockingIOError:
                break
            try:
                message = decode_message(payload)
            except ValueError as error:
                self._warning_callback(str(error))
                continue
            messages.append(message)
            self.last_received_at = received_at
        return messages

    def is_online(self, now: float | None = None, timeout: float = 2.0) -> bool:
        if self.last_received_at is None:
            return False
        current = time.monotonic() if now is None else float(now)
        return current - self.last_received_at <= timeout

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "UdpReceiver":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
