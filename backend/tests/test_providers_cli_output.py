from __future__ import annotations

from types import SimpleNamespace

from app import providers


def test_cli_output_looks_like_error_for_structured_json() -> None:
    assert providers._cli_output_looks_like_error('{"type":"result","is_error":true}')
    assert providers._cli_output_looks_like_error('{"subtype":"error_during_execution","is_error":true}')


def test_cli_output_looks_like_error_for_plain_error() -> None:
    assert providers._cli_output_looks_like_error("error: unexpected argument '--x' found")
    assert providers._cli_output_looks_like_error("ok", "error: permission denied")


def test_generate_with_claude_code_retries_when_stdout_is_structured_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENSIFT_CLAUDE_CODE_CMD", "claude")
    monkeypatch.setenv("OPENSIFT_CLAUDE_CODE_ARGS", "")
    monkeypatch.setattr(providers, "_which_cmd", lambda _cmd: "/bin/claude")

    calls = {"n": 0}

    def _fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(
                returncode=0,
                stdout='{"type":"result","subtype":"error_during_execution","is_error":true}',
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="usable answer", stderr="")

    monkeypatch.setattr(providers.subprocess, "run", _fake_run)
    out = providers.generate_with_claude_code("hello")
    assert out == "usable answer"
    assert calls["n"] >= 2


def test_generate_with_codex_skips_error_like_stdout(monkeypatch) -> None:
    monkeypatch.setenv("OPENSIFT_CODEX_CMD", "codex")
    monkeypatch.setenv("OPENSIFT_CODEX_ARGS", "")
    monkeypatch.setattr(providers, "build_codex_cli_invocations", lambda model=None: [["codex", "exec", "-"], ["codex", "exec", "-"]])
    monkeypatch.setattr(providers, "_codex_subprocess_env", lambda: {})

    calls = {"n": 0}

    def _fake_run_subprocess(args, stdin_text, timeout_s=180, env=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return 'error: unexpected argument "--skip-git-repo-check" found'
        return "codex usable answer"

    monkeypatch.setattr(providers, "_run_subprocess", _fake_run_subprocess)
    out = providers.generate_with_codex("hello")
    assert out == "codex usable answer"
    assert calls["n"] == 2
