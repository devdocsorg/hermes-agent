import json

import pytest

from gateway.codex_slack_bridge import (
    CodexSlackMapping,
    META_PREFIX,
    mapping_for_codex_thread,
    mapping_for_slack_thread,
    persist_mapping,
    persist_codex_thread,
    resolve_binding,
)
from gateway.config import Platform
from gateway.session import SessionSource


class FakeMetaStore:
    def __init__(self):
        self.values = {}

    def get_meta(self, key):
        return self.values.get(key)

    def set_meta(self, key, value):
        self.values[key] = value


@pytest.fixture(autouse=True)
def isolated_bridge_state(tmp_path, monkeypatch):
    state = tmp_path / "codex_slack_bridge_state.sqlite3"
    monkeypatch.setattr(
        "gateway.codex_slack_bridge.state_path",
        lambda: state,
    )
    return state


def _slack_source(**overrides):
    values = {
        "platform": Platform.SLACK,
        "scope_id": "T123",
        "chat_id": "C456",
        "thread_id": "1720000000.000100",
        "chat_type": "thread",
    }
    values.update(overrides)
    return SessionSource(**values)


def test_resolve_binding_uses_channel_project_and_restores_thread(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    registry = tmp_path / "codex_slack_bridge.json"
    registry.write_text(
        json.dumps({
            "channels": {
                "T123:C456": {"project_path": str(project)},
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gateway.codex_slack_bridge.registry_path",
        lambda: registry,
    )
    store = FakeMetaStore()
    meta_key = META_PREFIX + "slack:T123:C456:1720000000.000100"
    store.values[meta_key] = json.dumps({
        "codex_thread_id": "codex-thread-existing",
    })

    binding = resolve_binding(_slack_source(), store)

    assert binding is not None
    assert binding.cwd == str(project)
    assert binding.channel_key == "T123:C456"
    assert binding.codex_thread_id == "codex-thread-existing"


def test_persisted_thread_is_restored_for_same_slack_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    registry = tmp_path / "codex_slack_bridge.json"
    registry.write_text(
        json.dumps({"default_project_path": str(project)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gateway.codex_slack_bridge.registry_path",
        lambda: registry,
    )
    store = FakeMetaStore()
    source = _slack_source()
    binding = resolve_binding(source, store)

    persist_codex_thread(store, binding, "codex-thread-new")
    restored = resolve_binding(source, store)

    assert restored is not None
    assert restored.codex_thread_id == "codex-thread-new"
    saved = json.loads(store.values[META_PREFIX + binding.key])
    assert saved["cwd"] == str(project)
    assert saved["channel_key"] == "default"


def test_non_slack_or_top_level_message_is_not_bound(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    registry = tmp_path / "codex_slack_bridge.json"
    registry.write_text(
        json.dumps({"default_project_path": str(project)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gateway.codex_slack_bridge.registry_path",
        lambda: registry,
    )
    store = FakeMetaStore()

    assert (
        resolve_binding(
            _slack_source(platform=Platform.DISCORD),
            store,
        )
        is None
    )
    assert resolve_binding(_slack_source(thread_id=None), store) is None


def test_shared_mapping_restores_binding_without_hermes_meta(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    registry = tmp_path / "codex_slack_bridge.json"
    registry.write_text(
        json.dumps({
            "channels": {
                "T123:C456": {"project_path": str(project)},
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gateway.codex_slack_bridge.registry_path",
        lambda: registry,
    )
    persist_mapping(
        CodexSlackMapping(
            codex_thread_id="codex-shared",
            workspace_id="T123",
            channel_id="C456",
            root_ts="1720000000.000100",
            cwd=str(project),
            channel_key="T123:C456",
        )
    )

    binding = resolve_binding(_slack_source(), FakeMetaStore())

    assert binding is not None
    assert binding.codex_thread_id == "codex-shared"


def test_persist_codex_thread_writes_bidirectional_shared_mapping(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    registry = tmp_path / "codex_slack_bridge.json"
    registry.write_text(
        json.dumps({
            "channels": {
                "T123:C456": {"project_path": str(project)},
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gateway.codex_slack_bridge.registry_path",
        lambda: registry,
    )
    store = FakeMetaStore()
    binding = resolve_binding(_slack_source(), store)

    persist_codex_thread(store, binding, "codex-bidirectional")

    by_codex = mapping_for_codex_thread("codex-bidirectional")
    by_slack = mapping_for_slack_thread(
        "T123",
        "C456",
        "1720000000.000100",
    )
    assert by_codex is not None
    assert by_slack is not None
    assert by_codex == by_slack
