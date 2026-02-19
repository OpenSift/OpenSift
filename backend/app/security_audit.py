from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


SENSITIVE_FILE_MODE_MAX = 0o600
SENSITIVE_DIR_MODE_MAX = 0o700


@dataclass
class AuditFinding:
    severity: str  # info | warn | high
    check: str
    message: str
    path: str = ""
    recommended: str = ""


def _perm_bits(path: Path) -> Optional[int]:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _is_mode_too_open(actual: int, maximum: int) -> bool:
    return (actual & ~maximum) != 0


def _format_mode(mode: int) -> str:
    return oct(mode)


def _check_sensitive_file(path: Path, findings: List[AuditFinding], fix_perms: bool = False) -> None:
    mode = _perm_bits(path)
    if mode is None:
        findings.append(
            AuditFinding(
                severity="info",
                check="sensitive_file_exists",
                message="Sensitive file not present.",
                path=str(path),
            )
        )
        return
    if _is_mode_too_open(mode, SENSITIVE_FILE_MODE_MAX):
        fixed = False
        if fix_perms:
            try:
                os.chmod(path, SENSITIVE_FILE_MODE_MAX)
                new_mode = _perm_bits(path)
                fixed = bool(new_mode is not None and not _is_mode_too_open(new_mode, SENSITIVE_FILE_MODE_MAX))
            except Exception:
                fixed = False
        if fixed:
            findings.append(
                AuditFinding(
                    severity="warn",
                    check="sensitive_file_permissions_fixed",
                    message=f"Sensitive file permissions were too open ({_format_mode(mode)}) and were auto-fixed.",
                    path=str(path),
                    recommended="Verify ownership and keep file private.",
                )
            )
            return
        findings.append(
            AuditFinding(
                severity="high",
                check="sensitive_file_permissions",
                message=f"Sensitive file permissions are too open: {_format_mode(mode)}",
                path=str(path),
                recommended=f"chmod 600 {path}",
            )
        )
    else:
        findings.append(
            AuditFinding(
                severity="info",
                check="sensitive_file_permissions",
                message=f"Permissions are restricted: {_format_mode(mode)}",
                path=str(path),
            )
        )


def _check_sensitive_dir(path: Path, findings: List[AuditFinding], fix_perms: bool = False) -> None:
    mode = _perm_bits(path)
    if mode is None:
        findings.append(
            AuditFinding(
                severity="info",
                check="sensitive_dir_exists",
                message="Sensitive directory not present.",
                path=str(path),
            )
        )
        return
    if _is_mode_too_open(mode, SENSITIVE_DIR_MODE_MAX):
        fixed = False
        if fix_perms:
            try:
                os.chmod(path, SENSITIVE_DIR_MODE_MAX)
                new_mode = _perm_bits(path)
                fixed = bool(new_mode is not None and not _is_mode_too_open(new_mode, SENSITIVE_DIR_MODE_MAX))
            except Exception:
                fixed = False
        if fixed:
            findings.append(
                AuditFinding(
                    severity="warn",
                    check="sensitive_dir_permissions_fixed",
                    message=f"Sensitive directory permissions were too open ({_format_mode(mode)}) and were auto-fixed.",
                    path=str(path),
                    recommended="Verify ownership and keep directory private.",
                )
            )
            return
        findings.append(
            AuditFinding(
                severity="warn",
                check="sensitive_dir_permissions",
                message=f"Sensitive directory permissions are too open: {_format_mode(mode)}",
                path=str(path),
                recommended=f"chmod 700 {path}",
            )
        )
    else:
        findings.append(
            AuditFinding(
                severity="info",
                check="sensitive_dir_permissions",
                message=f"Permissions are restricted: {_format_mode(mode)}",
                path=str(path),
            )
        )


def _default_codex_auth_path(project_root: Path) -> Path:
    override = os.environ.get("OPENSIFT_CODEX_AUTH_PATH", "").strip()
    if override:
        return Path(override).expanduser()

    env_file = project_root / ".env"
    if env_file.exists():
        try:
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENSIFT_CODEX_AUTH_PATH" and value.strip():
                    return Path(value.strip()).expanduser()
        except Exception:
            pass
    return Path.home() / ".codex" / "auth.json"


def run_security_audit(project_root: Path, fix_perms: bool = False) -> Tuple[List[AuditFinding], int]:
    findings: List[AuditFinding] = []

    sensitive_files: Sequence[Path] = (
        project_root / ".env",
        project_root / ".opensift_auth.json",
        project_root / "SOUL.md",
        _default_codex_auth_path(project_root),
    )
    sensitive_dirs: Sequence[Path] = (
        project_root / ".opensift_sessions",
        project_root / ".opensift_library",
        project_root / ".opensift_quiz_attempts",
        project_root / ".opensift_flashcards",
        project_root / ".opensift_logs",
        project_root / ".chroma",
    )

    for p in sensitive_files:
        _check_sensitive_file(p, findings, fix_perms=fix_perms)

    for p in sensitive_dirs:
        _check_sensitive_dir(p, findings, fix_perms=fix_perms)

    geteuid = getattr(os, "geteuid", None)
    is_root = bool(callable(geteuid) and geteuid() == 0)
    if is_root:
        findings.append(
            AuditFinding(
                severity="warn",
                check="runtime_user",
                message="Process is running as root. Use a non-root user for local and container runtime.",
                recommended="Run OpenSift as a regular user.",
            )
        )
    else:
        findings.append(
            AuditFinding(
                severity="info",
                check="runtime_user",
                message="Runtime is not root.",
            )
        )

    if Path("/var/run/docker.sock").exists():
        findings.append(
            AuditFinding(
                severity="warn",
                check="docker_socket_exposed",
                message="Docker socket is accessible; this can grant broad host control if compromised.",
                path="/var/run/docker.sock",
                recommended="Avoid mounting docker.sock into containers unless strictly required.",
            )
        )

    if (os.environ.get("OPENSIFT_LOG_LEVEL", "") or "").strip().upper() == "DEBUG":
        findings.append(
            AuditFinding(
                severity="warn",
                check="debug_logging",
                message="DEBUG logging may increase metadata exposure in logs.",
                recommended="Prefer OPENSIFT_LOG_LEVEL=INFO for normal usage.",
            )
        )

    highs = sum(1 for f in findings if f.severity == "high")
    warns = sum(1 for f in findings if f.severity == "warn")
    exit_code = 1 if highs > 0 else (2 if warns > 0 else 0)
    return findings, exit_code


def format_audit_report(findings: Sequence[AuditFinding]) -> str:
    lines: List[str] = []
    lines.append("OpenSift Security Audit")
    lines.append("=" * 72)

    order = {"high": 0, "warn": 1, "info": 2}
    for f in sorted(findings, key=lambda x: (order.get(x.severity, 9), x.check, x.path)):
        prefix = {"high": "[HIGH]", "warn": "[WARN]", "info": "[INFO]"}.get(f.severity, "[INFO]")
        head = f"{prefix} {f.check}: {f.message}"
        lines.append(head)
        if f.path:
            lines.append(f"       path: {f.path}")
        if f.recommended:
            lines.append(f"       fix : {f.recommended}")

    highs = sum(1 for f in findings if f.severity == "high")
    warns = sum(1 for f in findings if f.severity == "warn")
    infos = sum(1 for f in findings if f.severity == "info")
    lines.append("-" * 72)
    lines.append(f"Summary: high={highs} warn={warns} info={infos}")
    return "\n".join(lines)
