"""Canonical DevDocs Memory policy for background self-improvement reviews.

The normal DevDocs MCP Memory tools keep their foreground semantics. This
module is activated only inside the background-review fork and adds the
mechanical constraints that an unattended writer needs:

* Personal ownership only.
* Concise, distilled, non-sensitive durable context only.
* Semantic search before capture.
* Deterministic IDs so retries cannot create duplicate records.

The active flag is a ContextVar so concurrent tool workers inherit it through
``tools.thread_context.propagate_context_to_thread``.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

CANONICAL_MEMORY_CAPTURE_TOOL = "mcp_devdocs_memory_capture"
CANONICAL_MEMORY_SEARCH_TOOL = "mcp_devdocs_memory_search"
CANONICAL_MEMORY_TOOLS = frozenset(
    {CANONICAL_MEMORY_CAPTURE_TOOL, CANONICAL_MEMORY_SEARCH_TOOL}
)

_AUTO_MEMORY_NAMESPACE = uuid.UUID("cf5e3e39-2d34-4c11-85ec-c6e3648d6b0d")
_AUTO_MEMORY_MAX_CHARS = 1200
_AUTO_MEMORY_DUPLICATE_SCORE = 0.90

_policy_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "background_review_canonical_memory_policy",
    default=False,
)

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk|ghp|github_pat|xox[abprs])[-_][A-Za-z0-9_-]{8,}"),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|"
        r"passwd|secret|authorization|cookie)\b\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)
_SENSITIVE_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:social security|credit card|bank account|routing number)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:medical diagnosis|health condition|sexual orientation|religious belief|"
        r"political affiliation)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
)
_RAW_CONTENT_PATTERNS = (
    re.compile(r"<untrusted_tool_result\b", re.IGNORECASE),
    re.compile(r"\b(?:raw transcript|verbatim transcript|source payload|imported payload)\b", re.IGNORECASE),
    re.compile(r"(?m)^\s*(?:USER|ASSISTANT|TOOL|SYSTEM)\s*:", re.IGNORECASE),
    re.compile(r"(?m)^\s*(?:From|To|Subject|Authorization|Cookie)\s*:", re.IGNORECASE),
)
_TRANSIENT_PATTERNS = (
    re.compile(
        r"\b(?:today|tomorrow|yesterday|this session|right now|temporarily|"
        r"currently failing|failed|failure|error|timeout|unavailable|"
        r"not working|missing credential|missing binary)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:asked|requested|wanted)\s+(?:me|the agent)\s+to\b", re.IGNORECASE),
)
_DURABLE_MARKERS = re.compile(
    r"\b(?:prefer(?:s|red)?|expect(?:s|ed)?|always|default|decision|decided|"
    r"working style|recurring|long[- ]term|responsible for|role is|uses|"
    r"wants (?:me|the agent) to|should)\b",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _normalized_identity(value: str) -> str:
    return _normalize_text(value).casefold()


def _looks_like_raw_json(body: str) -> bool:
    stripped = body.strip()
    if not stripped.startswith(("{", "[")):
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, (dict, list))


def auto_memory_block_reason(body: Any) -> Optional[str]:
    """Return a safe, content-free reason when an automatic capture is denied."""
    text = _normalize_text(body)
    if len(text) < 20:
        return "Automatic Memory capture requires a durable, non-trivial fact."
    if len(text) > _AUTO_MEMORY_MAX_CHARS:
        return "Automatic Memory capture must be a concise distilled fact."
    if "\x00" in str(body):
        return "Automatic Memory capture rejected invalid content."
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return "Automatic Memory capture cannot store credentials or secrets."
    if any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS):
        return "Automatic Memory capture cannot store sensitive personal data."
    if _looks_like_raw_json(str(body)) or any(
        pattern.search(str(body)) for pattern in _RAW_CONTENT_PATTERNS
    ):
        return "Automatic Memory capture cannot copy raw transcripts or source payloads."
    if len(str(body).splitlines()) > 4:
        return "Automatic Memory capture must distill source material into one concise fact."
    if any(pattern.search(text) for pattern in _TRANSIENT_PATTERNS) and not _DURABLE_MARKERS.search(text):
        return "Automatic Memory capture cannot store transient failures or one-off task state."
    return None


@contextlib.contextmanager
def canonical_memory_review_policy(enabled: bool):
    """Enable canonical-memory protections for one background review."""
    token = _policy_active.set(bool(enabled))
    try:
        yield
    finally:
        _policy_active.reset(token)


def canonical_memory_review_policy_active() -> bool:
    return bool(_policy_active.get())


def apply_canonical_memory_request_policy(
    tool_name: str,
    args: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Force Personal scope and deterministic capture metadata when active."""
    current = dict(args or {})
    if not canonical_memory_review_policy_active() or tool_name not in CANONICAL_MEMORY_TOOLS:
        return current

    for key in (
        "organization_id",
        "organizationId",
        "organization",
        "team_id",
        "teamId",
        "workspace_id",
        "workspaceId",
    ):
        current.pop(key, None)
    current["ownership"] = "personal"

    if tool_name == CANONICAL_MEMORY_SEARCH_TOOL:
        current["query"] = _normalize_text(current.get("query"))
        try:
            limit = int(current.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        current["limit"] = max(1, min(limit, 10))
        return current

    body = _normalize_text(current.get("body"))
    current["body"] = body
    current["memory_id"] = str(uuid.uuid5(_AUTO_MEMORY_NAMESPACE, _normalized_identity(body)))

    title = _normalize_text(current.get("title"))
    if title:
        current["title"] = title[:240]
    else:
        current.pop("title", None)

    tags = current.get("tags")
    safe_tags = []
    if isinstance(tags, list):
        for tag in tags:
            normalized = _normalize_text(tag)[:60]
            if normalized and normalized not in safe_tags:
                safe_tags.append(normalized)
    for tag in ("auto-learning", "background-review"):
        if tag not in safe_tags:
            safe_tags.append(tag)
    current["tags"] = safe_tags[:20]
    return current


def canonical_memory_policy_block_message(
    tool_name: str,
    args: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Block unsafe canonical writes while leaving foreground calls untouched."""
    if (
        not canonical_memory_review_policy_active()
        or tool_name != CANONICAL_MEMORY_CAPTURE_TOOL
    ):
        return None
    return auto_memory_block_reason((args or {}).get("body"))


def _json_value_from_text(value: str) -> Any:
    text = value.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    candidates = [index for index in (text.find("["), text.find("{")) if index >= 0]
    for index in sorted(candidates):
        try:
            return json.loads(text[index:])
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _parse_tool_result(result: Any) -> tuple[Any, Optional[str]]:
    if not isinstance(result, str):
        return None, "Memory tool returned a non-text result."
    try:
        outer = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None, "Memory tool returned malformed JSON."
    if not isinstance(outer, dict):
        return None, "Memory tool returned an unexpected result."
    error = outer.get("error")
    if isinstance(error, str) and error:
        return None, error

    structured = outer.get("structuredContent")
    if structured is not None:
        return structured, None
    payload = outer.get("result")
    if isinstance(payload, (dict, list)):
        return payload, None
    if isinstance(payload, str):
        parsed = _json_value_from_text(payload)
        if parsed is not None:
            return parsed, None
    return None, "Memory tool returned an unreadable payload."


def _iter_memory_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("items", "memories", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item


def _duplicate_record(records: Iterable[Dict[str, Any]], body: str) -> Optional[Dict[str, Any]]:
    identity = _normalized_identity(body)
    for record in records:
        if _normalized_identity(record.get("body", "")) == identity:
            return record
        score = record.get("score")
        if isinstance(score, (int, float)) and float(score) >= _AUTO_MEMORY_DUPLICATE_SCORE:
            return record
    return None


def dispatch_canonical_memory_tool(
    tool_name: str,
    args: Dict[str, Any],
    dispatch: Callable[[str, Dict[str, Any]], Any],
) -> Any:
    """Dispatch one tool call, adding search-before-capture when active."""
    if (
        not canonical_memory_review_policy_active()
        or tool_name != CANONICAL_MEMORY_CAPTURE_TOOL
    ):
        return dispatch(tool_name, args)

    safe_args = apply_canonical_memory_request_policy(tool_name, args)
    block_reason = auto_memory_block_reason(safe_args.get("body"))
    if block_reason:
        logger.warning("Background review Personal Memory capture blocked: %s", block_reason)
        return json.dumps({"error": block_reason}, ensure_ascii=False)

    memory_id = safe_args["memory_id"]
    search_args = apply_canonical_memory_request_policy(
        CANONICAL_MEMORY_SEARCH_TOOL,
        {"query": safe_args["body"], "limit": 5},
    )
    search_result = dispatch(CANONICAL_MEMORY_SEARCH_TOOL, search_args)
    search_payload, search_error = _parse_tool_result(search_result)
    if search_error:
        logger.warning(
            "Background review Personal Memory dedup search failed for memory_id=%s: %s",
            memory_id,
            search_error,
        )
        return json.dumps(
            {"error": "Automatic Personal Memory deduplication search failed; capture was skipped."},
            ensure_ascii=False,
        )

    duplicate = _duplicate_record(_iter_memory_records(search_payload), safe_args["body"])
    if duplicate is not None:
        logger.info(
            "Background review Personal Memory capture skipped as duplicate: "
            "memory_id=%s existing_memory_id=%s score=%s",
            memory_id,
            duplicate.get("id", ""),
            duplicate.get("score", ""),
        )
        return json.dumps(
            {
                "success": True,
                "skipped": True,
                "reason": "duplicate",
                "message": "Equivalent Personal Memory already exists.",
                "existing_memory_id": duplicate.get("id"),
            },
            ensure_ascii=False,
        )

    logger.info(
        "Background review Personal Memory capture starting: memory_id=%s",
        memory_id,
    )
    capture_result = dispatch(tool_name, safe_args)
    _payload, capture_error = _parse_tool_result(capture_result)
    if capture_error:
        logger.warning(
            "Background review Personal Memory capture failed for memory_id=%s: %s",
            memory_id,
            capture_error,
        )
    else:
        logger.info(
            "Background review Personal Memory capture completed: memory_id=%s",
            memory_id,
        )
    return capture_result


def canonical_memory_tools_available(agent: Any) -> bool:
    """Return True when both canonical Memory tools are usable by this agent."""
    visible_names = set(getattr(agent, "valid_tool_names", set()) or set())
    if CANONICAL_MEMORY_TOOLS.issubset(visible_names):
        return True
    # Avoid a registry walk for ordinary local-memory, skill-only, or
    # memory-disabled profiles. A hidden canonical tool can only be reachable
    # when Tool Search assembled the bridge surface.
    try:
        from tools.tool_search import BRIDGE_TOOL_NAMES

        if not (BRIDGE_TOOL_NAMES & visible_names):
            return False
    except Exception:
        return False

    try:
        from model_tools import get_tool_definitions

        definitions = get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        scoped_names = {
            (definition.get("function") or {}).get("name", "")
            for definition in definitions or []
        }
        return CANONICAL_MEMORY_TOOLS.issubset(scoped_names)
    except Exception:
        logger.debug("Could not resolve canonical Memory tool availability", exc_info=True)
        return False


def background_review_memory_allowed_tools(agent: Any) -> set[str]:
    """Return canonical/bridge names the review whitelist may safely expose."""
    if not canonical_memory_tools_available(agent):
        return set()

    allowed = set(CANONICAL_MEMORY_TOOLS)
    visible_names = set(getattr(agent, "valid_tool_names", set()) or set())
    try:
        from tools.tool_search import BRIDGE_TOOL_NAMES

        allowed.update(BRIDGE_TOOL_NAMES & visible_names)
    except Exception:
        pass
    return allowed


__all__ = [
    "CANONICAL_MEMORY_CAPTURE_TOOL",
    "CANONICAL_MEMORY_SEARCH_TOOL",
    "CANONICAL_MEMORY_TOOLS",
    "apply_canonical_memory_request_policy",
    "auto_memory_block_reason",
    "background_review_memory_allowed_tools",
    "canonical_memory_policy_block_message",
    "canonical_memory_review_policy",
    "canonical_memory_review_policy_active",
    "canonical_memory_tools_available",
    "dispatch_canonical_memory_tool",
]
