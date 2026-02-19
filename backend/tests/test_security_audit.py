from __future__ import annotations

import os
from pathlib import Path

from app.security_audit import format_audit_report, run_security_audit


def test_security_audit_flags_overly_permissive_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
    os.chmod(env_file, 0o644)

    monkeypatch.delenv("OPENSIFT_CODEX_AUTH_PATH", raising=False)
    findings, rc = run_security_audit(tmp_path, fix_perms=False)

    assert any(f.check == "sensitive_file_permissions" and f.path.endswith(".env") and f.severity == "high" for f in findings)
    assert rc == 1


def test_security_audit_fix_perms_restricts_mode(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
    os.chmod(env_file, 0o666)

    monkeypatch.delenv("OPENSIFT_CODEX_AUTH_PATH", raising=False)
    findings, _ = run_security_audit(tmp_path, fix_perms=True)
    _ = findings

    mode = env_file.stat().st_mode & 0o777
    assert mode == 0o600


def test_security_audit_report_format_contains_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENSIFT_CODEX_AUTH_PATH", raising=False)
    findings, _ = run_security_audit(tmp_path, fix_perms=False)
    report = format_audit_report(findings)
    assert "OpenSift Security Audit" in report
    assert "Summary:" in report

