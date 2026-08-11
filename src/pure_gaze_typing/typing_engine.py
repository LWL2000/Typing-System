from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PageKind(str, Enum):
    MAIN = "main"
    LETTERS = "letters"
    FUNCTIONS = "functions"


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
    requires_clear_confirmation: bool = False


class TypingEngine:
    _GROUPS = (
        tuple("ABCDE"),
        tuple("FGHIJ"),
        tuple("KLMNO"),
        tuple("PQRST"),
        tuple("UVWXYZ"),
    )

    def __init__(self, current_line: str = "") -> None:
        self.page_kind = PageKind.MAIN
        self.current_line = current_line
        self._history: list[str] = []
        self._group_index: int | None = None
        self._pending_clear = False
        self._pending_send_text: str | None = None
        self.paused = False

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def targets(self) -> tuple[TargetSpec, ...]:
        if self.paused:
            return (TargetSpec("function_resume", "继续", 4, "resume"),)
        if self.page_kind is PageKind.MAIN:
            labels = ("A-E", "F-J", "K-O", "P-T", "U-Z")
            targets = [
                TargetSpec(f"main_group_{index}", label, index, "open_letters")
                for index, label in enumerate(labels)
            ]
            targets.append(TargetSpec("main_functions", "功能", 5, "open_functions"))
            return tuple(targets)
        if self.page_kind is PageKind.LETTERS:
            assert self._group_index is not None
            letters = self._GROUPS[self._group_index]
            targets = [
                TargetSpec(f"letter_{letter}", letter, index, "type_letter")
                for index, letter in enumerate(letters)
            ]
            if len(targets) < 6:
                targets.append(TargetSpec("letter_space", "空格", 5, "space"))
            return tuple(targets)
        labels = (
            ("function_delete", "删除", "delete"),
            ("function_space", "空格", "space"),
            ("function_send", "发送", "send"),
            ("function_clear", "清空", "clear"),
            ("function_pause", "暂停", "pause"),
            ("function_return", "返回", "return"),
        )
        return tuple(
            TargetSpec(target_id, label, position, action)
            for position, (target_id, label, action) in enumerate(labels)
        )

    def activate(self, target_id: str) -> TypingEffect:
        target = next((item for item in self.targets() if item.target_id == target_id), None)
        if target is None:
            raise ValueError(f"当前页面不存在目标：{target_id}")
        if target_id != "function_clear":
            self._pending_clear = False
        action = target.action
        if action == "resume":
            self.paused = False
            return TypingEffect(action, page_changed=True)
        if action == "open_letters":
            self.page_kind = PageKind.LETTERS
            self._group_index = target.position
            return TypingEffect(action, page_changed=True)
        if action == "open_functions":
            self.page_kind = PageKind.FUNCTIONS
            self._group_index = None
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
        if action == "clear":
            if not self._pending_clear:
                self._pending_clear = True
                return TypingEffect(action, requires_clear_confirmation=True)
            self._pending_clear = False
            changed = bool(self.current_line)
            self.current_line = ""
            return TypingEffect(action, text_changed=changed)
        if action == "pause":
            self.paused = True
            return TypingEffect(action, page_changed=True)
        if action == "return":
            self.return_to_main()
            return TypingEffect(action, page_changed=True)
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
        self._pending_clear = False
        self.paused = False

    def resume(self) -> None:
        self.paused = False

    def full_text(self) -> str:
        lines = [*self._history]
        if self.current_line or not lines:
            lines.append(self.current_line)
        return "\n".join(lines)
