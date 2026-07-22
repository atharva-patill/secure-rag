from rich.console import Console

from secure_rag.header import SecureRagHeader


def _render_text(header, **kwargs):
    with header.console.capture() as capture:
        header.console.print(header.render(**kwargs))
    return capture.get().splitlines()


def test_active_header_shows_spinner_title_and_status():
    console = Console(width=44)
    header = SecureRagHeader(console)

    lines = _render_text(header, status="Initializing", active=True, spinner="⠋")

    assert lines[0].startswith("⠋ Secure RAG")
    assert lines[0].endswith("Initializing")
    assert len(lines[1]) == 44


def test_idle_header_stops_spinner_and_keeps_status_right_aligned():
    console = Console(width=44)
    header = SecureRagHeader(console)

    lines = _render_text(header, status="Ready", active=False, spinner="⠋")

    assert lines[0].startswith("Secure RAG")
    assert lines[0].endswith("Ready")
    assert "⠋" not in lines[0]
    assert lines[1] == "─" * 44


def test_active_divider_uses_subtle_sweep_without_changing_width():
    console = Console(width=44)
    header = SecureRagHeader(console)

    lines = _render_text(
        header,
        status="Generating",
        active=True,
        spinner="⠹",
        divider_phase=8,
    )

    assert len(lines[1]) == 44
    assert "━" in lines[1]
    assert lines[1].count("━") <= 3
