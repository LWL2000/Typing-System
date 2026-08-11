from pure_gaze_typing.typing_engine import PageKind, TypingEngine


def test_group_selection_types_letter_and_fixed_back_returns_main():
    engine = TypingEngine()
    engine.activate("main_group_0")
    assert [target.label for target in engine.targets()][:5] == list("ABCDE")
    engine.activate("letter_A")
    assert engine.current_line == "A"
    engine.return_to_main()
    assert engine.page_kind is PageKind.MAIN


def test_u_to_z_group_exposes_all_six_letters():
    engine = TypingEngine()
    engine.activate("main_group_4")
    assert [target.label for target in engine.targets()] == list("UVWXYZ")


def test_clear_requires_two_consecutive_clear_activations():
    engine = TypingEngine(current_line="TEST")
    engine.activate("main_functions")
    first = engine.activate("function_clear")
    assert first.requires_clear_confirmation
    assert engine.current_line == "TEST"
    engine.activate("function_clear")
    assert engine.current_line == ""


def test_send_does_not_clear_until_persistence_is_confirmed():
    engine = TypingEngine(current_line="HELLO")
    engine.activate("main_functions")
    effect = engine.activate("function_send")
    assert effect.sent_text == "HELLO"
    assert engine.current_line == "HELLO"
    engine.confirm_send()
    assert engine.current_line == ""
    assert engine.history == ("HELLO",)


def test_pause_exposes_only_resume_at_the_same_position():
    engine = TypingEngine()
    engine.activate("main_functions")
    engine.activate("function_pause")
    assert engine.paused
    targets = engine.targets()
    assert [(target.label, target.position) for target in targets] == [("继续", 4)]
    engine.activate("function_resume")
    assert not engine.paused
