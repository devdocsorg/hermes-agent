"""Regression coverage for macOS app-container search prompts."""

from pathlib import Path
from unittest.mock import MagicMock

import tools.file_operations as file_operations
from tools.file_operations import ShellFileOperations


def _mock_env():
    env = MagicMock()
    env.cwd = "/"

    def execute(command, **_kwargs):
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if "command -v rg" in command:
            return {"output": "yes\n", "returncode": 0}
        if "command -v find" in command or "command -v grep" in command:
            return {"output": "yes\n", "returncode": 0}
        return {"output": "", "returncode": 1}

    env.execute.side_effect = execute
    return env


def _force_macos_roots(monkeypatch, home: Path):
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    monkeypatch.setattr(
        file_operations,
        "_MACOS_TCC_SEARCH_ROOTS",
        (
            str(home / "Library" / "Containers"),
            str(home / "Library" / "Group Containers"),
        ),
    )


def _executed_commands(env):
    return [call.args[0] for call in env.execute.call_args_list]


def test_direct_search_inside_app_container_is_blocked_before_shell(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    ops = ShellFileOperations(env)

    result = ops.search(
        "token",
        path=str(home / "Library" / "Containers" / "com.example.app"),
    )

    assert result.error is not None
    assert "macOS protects app-container data" in result.error
    env.execute.assert_not_called()


def test_broad_home_search_is_blocked_before_shell(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    ops = ShellFileOperations(env)

    result = ops.search("needle", path=str(home))

    assert result.error is not None
    assert "scope is broad enough" in result.error
    env.execute.assert_not_called()


def test_relative_search_uses_session_cwd_for_scope_guard(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    env.cwd = str(home)
    ops = ShellFileOperations(env)

    result = ops.search("needle", path=".")

    assert result.error is not None
    assert "scope is broad enough" in result.error
    env.execute.assert_not_called()


def test_content_search_backend_prunes_app_container_roots_from_rg(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    ops = ShellFileOperations(env)

    ops._search_with_rg(
        "needle",
        str(home),
        file_glob=None,
        limit=50,
        offset=0,
        output_mode="content",
        context=0,
    )

    rg_command = next(
        command
        for command in _executed_commands(env)
        if command.startswith("set -o pipefail; rg ")
    )
    assert "--glob '!Library/Containers/**'" in rg_command
    assert "--glob '!Library/Group Containers/**'" in rg_command


def test_file_search_backend_prunes_app_container_roots_from_rg(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    ops = ShellFileOperations(env)

    ops._search_files_rg("*.py", str(home), limit=50, offset=0)

    rg_command = next(
        command
        for command in _executed_commands(env)
        if command.startswith("rg --files ")
    )
    assert "-g '!Library/Containers/**'" in rg_command
    assert "-g '!Library/Group Containers/**'" in rg_command


def test_find_fallback_prunes_exact_app_container_roots(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    ops = ShellFileOperations(env)
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")

    ops._search_files("*.py", str(home), limit=50, offset=0)

    find_commands = [command for command in _executed_commands(env) if command.startswith("find ")]
    assert find_commands
    for command in find_commands:
        assert f"-path '{home}/Library/Containers'" in command
        assert f"-path '{home}/Library/Group Containers'" in command
        assert "-prune -o" in command


def test_grep_fallback_excludes_container_basenames_only_for_broad_root(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    ops = ShellFileOperations(env)

    ops._search_with_grep(
        "needle",
        str(home),
        file_glob=None,
        limit=50,
        offset=0,
        output_mode="content",
        context=0,
    )

    grep_command = next(
        command
        for command in _executed_commands(env)
        if command.startswith("set -o pipefail; grep ")
    )
    assert "--exclude-dir='Containers'" in grep_command
    assert "--exclude-dir='Group Containers'" in grep_command


def test_normal_project_search_has_no_tcc_exclusions(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "projects" / "app"
    _force_macos_roots(monkeypatch, home)
    env = _mock_env()
    ops = ShellFileOperations(env)

    ops.search("needle", path=str(project))

    rg_command = next(
        command
        for command in _executed_commands(env)
        if command.startswith("set -o pipefail; rg ")
    )
    assert "Library/Containers" not in rg_command
    assert "Library/Group Containers" not in rg_command
