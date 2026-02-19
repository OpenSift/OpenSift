from __future__ import annotations

import itertools

import cli_chat
import opensift
import pytest
import ui_app


def _clear_provider_env(monkeypatch) -> None:
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CHATGPT_CODEX_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)


def test_ui_preferred_provider_combinations(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(ui_app, "codex_auth_detected", lambda: False)
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: False)

    monkeypatch.setattr(ui_app, "claude_code_cli_available", lambda: True)
    assert ui_app._preferred_provider_default() == "claude_code"

    monkeypatch.setattr(ui_app, "claude_code_cli_available", lambda: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert ui_app._preferred_provider_default() == "claude"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ui_app, "codex_auth_detected", lambda: True)
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: True)
    assert ui_app._preferred_provider_default() == "codex"

    monkeypatch.setattr(ui_app, "codex_auth_detected", lambda: False)
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert ui_app._preferred_provider_default() == "openai"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "x")
    assert ui_app._preferred_provider_default() == "claude_code"


def test_launcher_default_provider_combinations(monkeypatch) -> None:
    monkeypatch.setattr(opensift, "_codex_auth_file_token", lambda: "")

    env = {}
    monkeypatch.setattr(opensift, "_claude_code_cli_available", lambda d: True)
    assert opensift._choose_default_provider(env) == "claude_code"

    monkeypatch.setattr(opensift, "_claude_code_cli_available", lambda d: False)
    env = {"ANTHROPIC_API_KEY": "x"}
    assert opensift._choose_default_provider(env) == "claude"

    env = {"CHATGPT_CODEX_OAUTH_TOKEN": "x"}
    assert opensift._choose_default_provider(env) == "codex"

    env = {"OPENAI_API_KEY": "x"}
    assert opensift._choose_default_provider(env) == "openai"

    env = {"CLAUDE_CODE_OAUTH_TOKEN": "x"}
    assert opensift._choose_default_provider(env) == "claude_code"


def test_cli_claude_code_fallback_combinations(monkeypatch) -> None:
    cfg = cli_chat.ChatConfig(provider="claude_code", model="", thinking_enabled=True)

    monkeypatch.setattr(cli_chat, "claude_code_cli_available", lambda: True)
    monkeypatch.setattr(cli_chat, "generate_with_claude_code", lambda prompt, model=None: "via-cli")
    assert cli_chat._run_generate(cfg, "q") == "via-cli"

    monkeypatch.setattr(cli_chat, "claude_code_cli_available", lambda: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(
        cli_chat,
        "generate_with_claude",
        lambda prompt, model=None, thinking_enabled=False, thinking_level="medium": "via-api",
    )
    assert cli_chat._run_generate(cfg, "q") == "via-api"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        cli_chat._run_generate(cfg, "q")
        assert False, "expected RuntimeError when no CLI and no Anthropic API key"
    except RuntimeError as e:
        assert "Claude Code CLI not found" in str(e)


@pytest.mark.parametrize(
    "claude_cli,anthropic,codex_detected,openai,claude_token,expected",
    [
        # Claude Code CLI always wins for default selection.
        (True, False, False, False, False, "claude_code"),
        (True, True, True, True, True, "claude_code"),
        # Without Claude Code CLI: Anthropic wins.
        (False, True, True, True, True, "claude"),
        (False, True, False, False, False, "claude"),
        # Next: Codex auth wins over OpenAI.
        (False, False, True, True, True, "codex"),
        (False, False, True, False, False, "codex"),
        # Then OpenAI.
        (False, False, False, True, True, "openai"),
        # Then Claude token fallback.
        (False, False, False, False, True, "claude_code"),
        # Nothing configured.
        (False, False, False, False, False, "claude_code"),
    ],
)
def test_ui_preferred_provider_priority_matrix(
    monkeypatch,
    claude_cli: bool,
    anthropic: bool,
    codex_detected: bool,
    openai: bool,
    claude_token: bool,
    expected: str,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(ui_app, "claude_code_cli_available", lambda: claude_cli)
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: codex_detected)
    monkeypatch.setattr(ui_app, "codex_auth_detected", lambda: codex_detected)
    if anthropic:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    if openai:
        monkeypatch.setenv("OPENAI_API_KEY", "x")
    if claude_token:
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "x")

    assert ui_app._preferred_provider_default() == expected


def test_launcher_provider_priority_full_boolean_matrix(monkeypatch) -> None:
    # Exhaustive 2^5 matrix over provider presence signals.
    for claude_cli, anthropic, codex_env, codex_file, openai, claude_token in itertools.product(
        (False, True), repeat=6
    ):
        env = {}
        if anthropic:
            env["ANTHROPIC_API_KEY"] = "x"
        if codex_env:
            env["CHATGPT_CODEX_OAUTH_TOKEN"] = "x"
        if openai:
            env["OPENAI_API_KEY"] = "x"
        if claude_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = "x"

        monkeypatch.setattr(opensift, "_claude_code_cli_available", lambda d, v=claude_cli: v)
        monkeypatch.setattr(opensift, "_codex_auth_file_token", lambda v=codex_file: ("x" if v else ""))

        got = opensift._choose_default_provider(env)
        if claude_cli:
            assert got == "claude_code"
        elif anthropic:
            assert got == "claude"
        elif codex_env or codex_file:
            assert got == "codex"
        elif openai:
            assert got == "openai"
        elif claude_token:
            assert got == "claude_code"
        else:
            assert got == "claude_code"


def test_ui_run_generate_explicit_provider_routing(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(ui_app, "generate_with_openai", lambda prompt, model=None: calls.append("openai") or "o")
    monkeypatch.setattr(ui_app, "generate_with_codex", lambda prompt, model=None: calls.append("codex") or "c")
    monkeypatch.setattr(
        ui_app,
        "generate_with_claude",
        lambda prompt, model=None, thinking_enabled=False, thinking_level="medium": calls.append("claude") or "a",
    )
    monkeypatch.setattr(ui_app, "generate_with_claude_code", lambda prompt, model=None: calls.append("claude_code") or "cc")

    monkeypatch.setattr(ui_app, "claude_code_cli_available", lambda: False)
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    assert ui_app._run_generate("openai", "q", "") == "o"
    assert ui_app._run_generate("codex", "q", "") == "c"
    assert ui_app._run_generate("claude", "q", "", thinking_enabled=True) == "a"
    # Explicit claude_code should fallback to claude API when CLI is unavailable.
    assert ui_app._run_generate("claude_code", "q", "", thinking_enabled=True) == "a"

    assert calls == ["openai", "codex", "claude", "claude"]


def test_ui_provider_model_discrepancy_resolution(monkeypatch) -> None:
    p, m, note = ui_app._resolve_provider_model_pair("openai", "claude-sonnet-4-6")
    assert p == "claude"
    assert m == "claude-sonnet-4-6"
    assert "Claude-family" in note

    monkeypatch.setattr(ui_app, "codex_auth_detected", lambda: True)
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: True)
    p2, m2, note2 = ui_app._resolve_provider_model_pair("claude", "gpt-5.3-codex")
    assert p2 == "codex"
    assert m2 == "gpt-5.3-codex"
    assert "GPT-family" in note2


def test_cli_provider_model_discrepancy_resolution(monkeypatch) -> None:
    p, m, _note = cli_chat._resolve_provider_model_pair("codex", "claude-sonnet-4-5")
    assert p == "claude"
    assert m == "claude-sonnet-4-5"

    monkeypatch.setattr(cli_chat, "codex_cli_available", lambda: True)
    monkeypatch.setattr(cli_chat, "codex_auth_detected", lambda: True)
    p2, m2, _note2 = cli_chat._resolve_provider_model_pair("claude_code", "gpt-5.3-codex")
    assert p2 == "codex"
    assert m2 == "gpt-5.3-codex"


def test_ui_run_generate_falls_back_from_codex_to_openai(monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(ui_app, "generate_with_openai", lambda prompt, model=None: "openai-fallback")
    out = ui_app._run_generate("codex", "hello", model="")
    assert out == "openai-fallback"


def test_cli_run_generate_falls_back_from_codex_to_openai(monkeypatch) -> None:
    cfg = cli_chat.ChatConfig(provider="codex", model="")
    monkeypatch.setattr(cli_chat, "codex_cli_available", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(cli_chat, "generate_with_openai", lambda prompt, model=None: "openai-fallback")
    out = cli_chat._run_generate(cfg, "hello")
    assert out == "openai-fallback"
