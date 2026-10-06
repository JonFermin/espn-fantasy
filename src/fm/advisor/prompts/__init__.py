"""The workers' prompt texts, one Markdown file per worker, read with :func:`prompt_text`.

A prompt is the stable prefix of every call its worker makes, so it is sent as a cached system block
(:class:`fm.advisor.client.Prompt`). Keeping the texts as files beside the code keeps them diffable and out of the
Python, and a worker reads its own by name: ``prompt_text("news_triage")`` is ``news_triage.md``.
"""

from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def prompt_text(name: str) -> str:
    """The text of ``<name>.md`` in this package, stripped; ``FileNotFoundError`` names the missing file."""
    path = resources.files(__name__).joinpath(f"{name}.md")
    if not path.is_file():
        raise FileNotFoundError(f"no prompt {name!r}: expected {name}.md in {__name__}")
    return path.read_text(encoding="utf-8").strip()
