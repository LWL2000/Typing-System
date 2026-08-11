from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PageKind(str, Enum):
    MAIN = "main"
    SUBMENU = "submenu"


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    label: str
    position: int
    action: str


@dataclass(frozen=True)
class TypingEffect:
    action: str
    text_changed: bool = False
    page_changed: bool = False
    sent_text: str | None = None


class TypingEngine:
    _GROUPS = (
        tuple("ABCDE"),
        tuple("FGHIJ"),
        tuple("KLMNO"),
        tuple("PQRST"),
        tuple("UVWXYZ"),
    )
    _MAIN_LABELS = (
        "A B C\nD E",
        "F G H\nI J",
        "K L M\nN O",
        "P Q R\nS T",
        "U V W\nX Y Z",
    )

    def __init__(self, current_line: str = "") -> None:
        self.page_kind = PageKind.MAIN
        self.current_line = current_line
        self._history: list[str] = []
        self._group_index: int | None = None
        self._pending_send_text: str | None = None

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def targets(self) -> tuple[TargetSpec, ...]:
        if self.page_kind is PageKind.MAIN:
            targets = [
                TargetSpec(f"main_group_{index}", label, index, "open_letters")
                for index, label in enumerate(self._MAIN_LABELS)
            ]
            targets.extend(
                (
                    TargetSpec("main_delete", "删除", 5, "delete"),
                    TargetSpec("main_space", "空格", 6, "space"),
                    TargetSpec("main_send", "发送", 7, "send"),
                )
            )
            return tuple(targets)

        assert self._group_index is not None
        letters = self._GROUPS[self._group_index]
        targets = [
            TargetSpec(f"letter_{letter}", letter, index, "type_letter")
            for index, letter in enumerate(letters)
        ]
        if len(targets) < 6:
            targets.append(TargetSpec("letter_space", "空格", 5, "space"))
        return tuple(targets)

    def activate(self, target_id: str) -> TypingEffect:
        target = next((item for item in self.targets() if item.target_id == target_id), None)
        if target is None:
            raise ValueError(f"当前页面不存在目标：{target_id}")
        action = target.action
        if action == "open_letters":
            self.page_kind = PageKind.SUBMENU
            self._group_index = target.position
            return TypingEffect(action, page_changed=True)
        if action == "type_letter":
            self.current_line += target.label
            return TypingEffect(action, text_changed=True)
        if action == "space":
            self.current_line += " "
            return TypingEffect(action, text_changed=True)
        if action == "delete":
            before = self.current_line
            self.current_line = self.current_line[:-1]
            return TypingEffect(action, text_changed=self.current_line != before)
        if action == "send":
            text = self.full_text()
            self._pending_send_text = text
            return TypingEffect(action, sent_text=text)
        raise RuntimeError(f"unsupported typing action: {action}")

    def confirm_send(self) -> None:
        if self._pending_send_text is None:
            raise RuntimeError("no send is pending")
        self._history.append(self.current_line)
        self.current_line = ""
        self._pending_send_text = None

    def return_to_main(self) -> None:
        self.page_kind = PageKind.MAIN
        self._group_index = None

    def full_text(self) -> str:
        return "\n".join((*self._history, self.current_line))
