from __future__ import annotations

import opensift


def test_non_fatal_exit_policy_for_mcp() -> None:
    assert opensift._is_non_fatal_service_exit("mcp", 0)
    assert not opensift._is_non_fatal_service_exit("mcp", 1)
    assert not opensift._is_non_fatal_service_exit("ui", 0)
