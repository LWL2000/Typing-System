from pure_gaze_typing.typing_engine import PageKind, TypingEngine


def test_main_menu_matches_reference_eight_target_keyboard():
    engine = TypingEngine()

    assert [(target.label, target.position) for target in engine.targets()] == [
        ("A B C\nD E", 0),
        ("F G H\nI J", 1),
        ("K L M\nN O", 2),
        ("P Q R\nS T", 3),
        ("U V W\nX Y Z", 4),
        ("删除", 5),
        ("空格", 6),
        ("发送", 7),
    ]


def test_group_selection_opens_six_target_submenu_and_types_letter():
    engine = TypingEngine()

    engine.activate("main_group_1")

    assert engine.page_kind is PageKind.SUBMENU
    assert [target.label for target in engine.targets()] == ["F", "G", "H", "I", "J", "空格"]
    engine.activate("letter_H")
    assert engine.current_line == "H"
    assert engine.page_kind is PageKind.SUBMENU


def test_u_to_z_group_exposes_all_six_letters():
    engine = TypingEngine()
    engine.activate("main_group_4")
    assert [target.label for target in engine.targets()] == list("UVWXYZ")


def test_delete_and_space_are_available_directly_from_main_menu():
    engine = TypingEngine(current_line="TEST")

    engine.activate("main_delete")
    engine.activate("main_space")

    assert engine.current_line == "TES "


def test_send_does_not_clear_until_persistence_is_confirmed():
    engine = TypingEngine(current_line="HELLO")

    effect = engine.activate("main_send")

    assert effect.sent_text == "HELLO"
    assert engine.current_line == "HELLO"
    engine.confirm_send()
    assert engine.current_line == ""
    assert engine.history == ("HELLO",)
    assert engine.full_text() == "HELLO\n"


def test_return_to_main_keeps_typed_text():
    engine = TypingEngine()
    engine.activate("main_group_0")
    engine.activate("letter_A")

    engine.return_to_main()

    assert engine.page_kind is PageKind.MAIN
    assert engine.current_line == "A"
