"""The advisor over the Claude Code CLI: workers run on the user's Claude subscription instead of an API key
(``[llm] backend = "claude_code"``, DESIGN section 10).

:class:`ClaudeCliTransport` is a :class:`fm.advisor.client.Transport`, so :class:`fm.advisor.client.AdvisorClient`
and every worker are unchanged: the budget check, the ``llm_usage`` record, the status rules. One call is one
headless ``claude -p`` run:

- the system blocks go in a ``--system-prompt-file`` (replacing Claude Code's own system prompt) and the question on
  stdin, so a long league context never meets the Windows command-line limit;
- the output type's schema goes in ``--json-schema``, and the answer is read from the run's ``structured_output``;
- ``--model`` and ``--effort`` come from the prompt, as with the SDK;
- the run is shut off from everything but the question: no tools (``--tools ""``) unless the prompt carries the
  web search server tool, which becomes Claude Code's ``WebSearch`` (its domains and use cap are stated in the system
  prompt, since the CLI takes neither as a flag), no MCP servers (``--strict-mcp-config``), no settings files
  (``--setting-sources ""``), no saved session, and an empty temporary working directory, so no ``CLAUDE.md`` is
  found;
- ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN`` are removed from the child's environment, so the CLI uses the
  ``claude login`` subscription and never bills an API key.

The run's events (``--output-format stream-json --verbose``) go to a file. The ``WebSearch`` results' links become
``search_urls``, so a close call is still judged against the URLs the search actually returned. The final ``result``
event gives the tokens (``modelUsage`` summed over the models that ran), the model that answered, and the answer. The
cost recorded is what the tokens would cost at the API list price, so ``[llm] daily_budget_usd`` still caps how much
of the subscription the advisor uses in a day.

The run is a :func:`fm.jobs.proctree.spawn` tree with a hard timeout (:data:`CLI_TIMEOUT_S`). A timeout, a CLI that
cannot start, an error result (a usage limit, a login that expired) or a run with no result raise
:class:`fm.advisor.client.AdvisorError`, which every caller already treats as "no advisor answer". The CLI has no
Message Batches API: the batch methods raise the same error. Tests pass ``command`` (a fake CLI script), never the
real ``claude`` (CLAUDE.md).
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, TypeAdapter, ValidationError

from fm.advisor.client import (
    AdvisorError,
    BatchRequest,
    BatchResult,
    BatchStatus,
    CallParams,
    RawReply,
    TokenUsage,
    Transport,
    output_format,
)
from fm.jobs import proctree

logger = logging.getLogger(__name__)

CLI_TIMEOUT_S: Final = 300.0
"""The longest one ``claude -p`` run may take before its tree is killed (a searched close call takes about a
minute)."""

CLI_SEARCH_TOOL: Final = "WebSearch"
"""Claude Code's web search tool, which stands in for the API's ``web_search`` server tool."""

STRIPPED_ENV: Final = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
"""Removed from the child's environment: the API credentials (the run must use the subscription login) and the
markers of an enclosing Claude Code session."""

NO_BATCHES: Final = 'the claude_code backend has no Message Batches API; set [llm] backend = "api" to batch'

type Spawn = Callable[..., proctree.ProcessTree]


def cli_env(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without :data:`STRIPPED_ENV`."""
    return {name: value for name, value in environ.items() if name not in STRIPPED_ENV}


def _search_tool(params: CallParams) -> Mapping[str, Any] | None:
    """The prompt's web search server tool, if any; another server tool has no CLI counterpart and is dropped."""
    found: Mapping[str, Any] | None = None
    for tool in params.get("tools", ()):
        if str(tool.get("type", "")).startswith("web_search"):
            found = tool
        else:
            logger.warning("advisor: claude CLI: server tool %r has no CLI counterpart; dropped", tool.get("type"))
    return found


def system_text(params: CallParams) -> str:
    """The system blocks joined, plus how to search when the prompt carries the web search tool."""
    parts = [block["text"] for block in params["system"]]
    tool = _search_tool(params)
    if tool is not None:
        rule = f"You have the {CLI_SEARCH_TOOL} tool."
        domains = [str(domain) for domain in tool.get("allowed_domains") or ()]
        if domains:
            rule += f" Search only these domains, passing them as allowed_domains: {', '.join(domains)}."
        if tool.get("max_uses"):
            rule += f" Search at most {int(tool['max_uses'])} times."
        parts.append(rule)
    return "\n\n".join(parts)


def user_text(params: CallParams) -> str:
    """The question: the text of the user turn's blocks."""
    texts: list[str] = []
    for message in params["messages"]:
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
            continue
        texts.extend(str(block["text"]) for block in content if isinstance(block, Mapping) and "text" in block)
    return "\n\n".join(texts)


def cli_argv(command: Sequence[str], params: CallParams, output: type[BaseModel], system_file: Path) -> list[str]:
    """The ``claude -p`` command line for one call (see the module docs)."""
    schema = json.dumps(output_format(output)["schema"], separators=(",", ":"))
    search = _search_tool(params) is not None
    argv = [
        *command,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        params["model"],
        "--system-prompt-file",
        str(system_file),
        "--json-schema",
        schema,
        "--tools",
        CLI_SEARCH_TOOL if search else "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--no-session-persistence",
    ]
    effort = params["output_config"].get("effort")
    if effort:
        argv += ["--effort", str(effort)]
    if search:
        argv += ["--allowedTools", CLI_SEARCH_TOOL]
    return argv


# --- reading the run --------------------------------------------------------------------------------------------------


def _links(text: str) -> list[str]:
    """The URLs of a ``WebSearch`` result: the JSON list after ``Links:`` (several when it searched more than once)."""
    urls: list[str] = []
    decoder = json.JSONDecoder()
    start = text.find("Links:")
    while start != -1:
        at = text.find("[", start)
        if at == -1:
            break
        try:
            links, end = decoder.raw_decode(text, at)
        except json.JSONDecodeError:
            start = text.find("Links:", at)
            continue
        if isinstance(links, list):
            urls.extend(str(link["url"]) for link in links if isinstance(link, Mapping) and link.get("url"))
        start = text.find("Links:", end)
    return urls


def _result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, Mapping))
    return ""


def _blocks(event: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    message = event.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return [block for block in content if isinstance(block, Mapping)] if isinstance(content, list) else []


def _usage(result: Mapping[str, Any]) -> tuple[TokenUsage, list[str]]:
    """The run's tokens and the models that ran, from ``modelUsage`` (every turn, every model), else ``usage``."""
    by_model = result.get("modelUsage")
    if isinstance(by_model, Mapping) and by_model:
        rows = [row for row in by_model.values() if isinstance(row, Mapping)]

        def total(key: str) -> int:
            return int(math.fsum(float(row.get(key) or 0) for row in rows))

        usage = TokenUsage(
            input_tokens=total("inputTokens"),
            output_tokens=total("outputTokens"),
            cache_creation_input_tokens=total("cacheCreationInputTokens"),
            cache_read_input_tokens=total("cacheReadInputTokens"),
        )
        return usage, [str(model) for model in by_model]
    raw = result.get("usage")
    raw = raw if isinstance(raw, Mapping) else {}
    usage = TokenUsage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(raw.get("cache_read_input_tokens") or 0),
    )
    return usage, []


def reply_from_stream(text: str, output: type[BaseModel], *, model: str) -> RawReply:
    """A :class:`RawReply` from a run's ``stream-json`` events (``model`` is the one asked for). Raises
    :class:`AdvisorError` for a run with no ``result`` event or an error result that is not a refusal."""
    searches: set[str] = set()
    urls: list[str] = []
    result: Mapping[str, Any] | None = None
    other: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            other.append(line)  # the CLI's stderr shares the file
            continue
        if not isinstance(event, Mapping):
            continue
        kind = event.get("type")
        if kind == "assistant":
            for block in _blocks(event):
                if block.get("type") == "tool_use" and block.get("name") == CLI_SEARCH_TOOL:
                    searches.add(str(block.get("id")))
        elif kind == "user":
            for block in _blocks(event):
                if block.get("type") == "tool_result" and str(block.get("tool_use_id")) in searches:
                    urls.extend(url for url in _links(_result_text(block.get("content"))) if url not in urls)
        elif kind == "result":
            result = event
    if result is None:
        tail = " / ".join(other[-3:]) or "no output"
        raise AdvisorError(f"claude CLI gave no result ({tail})")
    usage, models = _usage(result)
    served = model if model in models or not models else models[0]
    request_id = str(result.get("session_id")) if result.get("session_id") else None
    base = RawReply(request_id=request_id, model=served, stop_reason=None, usage=usage, fallback=served != model)
    if result.get("stop_reason") == "refusal":
        return _with(base, stop_reason="refusal", detail=str(result.get("result") or "") or None)
    if result.get("is_error") or result.get("subtype") != "success":
        what = result.get("result") or result.get("api_error_status") or "no detail"
        raise AdvisorError(f"claude CLI run failed ({result.get('subtype')}): {what}")
    structured = result.get("structured_output")
    if structured is None:
        return _with(base, stop_reason="end_turn", detail="the CLI returned no structured output", urls=urls)
    try:
        parsed = TypeAdapter(output).validate_python(structured)
    except ValidationError as exc:
        detail = f"output does not match {output.__name__} ({exc.error_count()} errors)"
        return _with(base, stop_reason="end_turn", detail=detail, urls=urls)
    return _with(base, stop_reason="end_turn", output=parsed, urls=urls)


def _with(
    base: RawReply,
    *,
    stop_reason: str,
    output: object = None,
    detail: str | None = None,
    urls: Sequence[str] = (),
) -> RawReply:
    return RawReply(
        request_id=base.request_id,
        model=base.model,
        stop_reason=stop_reason,
        usage=base.usage,
        output=output,
        detail=detail,
        fallback=base.fallback,
        search_urls=tuple(urls),
    )


# --- the transport ----------------------------------------------------------------------------------------------------


class ClaudeCliTransport(Transport):
    """Workers over ``claude -p`` on the subscription login (see the module docs). ``command`` is the CLI (its full
    path, or a fake in tests); ``spawn`` starts the run (:func:`fm.jobs.proctree.spawn`)."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = CLI_TIMEOUT_S,
        environ: Mapping[str, str] | None = None,
        spawn: Spawn = proctree.spawn,
    ) -> None:
        if not command:
            raise ValueError("a claude CLI command is required")
        self.command = tuple(str(part) for part in command)
        self.timeout = timeout
        self._environ = environ
        self._spawn = spawn

    def parse(self, params: CallParams, output: type[BaseModel], *, fallback: bool = False) -> RawReply:
        """One run. ``fallback`` has no CLI counterpart and is ignored (the CLI picks its own fallback model, which
        the reply's ``model`` then names)."""
        with tempfile.TemporaryDirectory(prefix="fm-claude-", ignore_cleanup_errors=True) as tmp:
            folder = Path(tmp)
            system_file = folder / "system.md"
            system_file.write_text(system_text(params), encoding="utf-8")
            question = folder / "question.md"
            question.write_text(user_text(params), encoding="utf-8")
            events = folder / "events.jsonl"
            argv = cli_argv(self.command, params, output, system_file)
            env = cli_env(os.environ if self._environ is None else self._environ)
            out_fd = os.open(events, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            in_fd = os.open(question, os.O_RDONLY)
            try:
                tree = self._spawn(argv, env=env, output=out_fd, cwd=folder, stdin=in_fd)
            except OSError as exc:
                raise AdvisorError(f"claude CLI could not start ({self.command[0]}): {exc}") from exc
            finally:
                os.close(out_fd)
                os.close(in_fd)
            try:
                code = tree.wait(self.timeout)
            except subprocess.TimeoutExpired:
                notes = tree.kill()
                trouble = f" ({'; '.join(notes)})" if notes else ""
                raise AdvisorError(f"claude CLI run timed out after {self.timeout:.0f}s; killed{trouble}") from None
            text = events.read_text(encoding="utf-8", errors="replace")
        try:
            return reply_from_stream(text, output, model=params["model"])
        except AdvisorError as exc:
            raise AdvisorError(f"{exc} (exit {code})") from None

    def create_batch(self, requests: Sequence[BatchRequest]) -> str:
        raise AdvisorError(NO_BATCHES)

    def batch_status(self, batch_id: str) -> BatchStatus:
        raise AdvisorError(NO_BATCHES)

    def batch_results(self, batch_id: str, output: type[BaseModel]) -> list[BatchResult]:
        raise AdvisorError(NO_BATCHES)


__all__ = [
    "CLI_SEARCH_TOOL",
    "CLI_TIMEOUT_S",
    "NO_BATCHES",
    "STRIPPED_ENV",
    "ClaudeCliTransport",
    "cli_argv",
    "cli_env",
    "reply_from_stream",
    "system_text",
    "user_text",
]
