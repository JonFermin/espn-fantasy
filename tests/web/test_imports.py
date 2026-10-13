"""The dashboard has no path to ESPN's write side (CLAUDE.md: only ``fm.executor``, through ``fm.browser``, writes).

Mirrors the checks ``tests/test_views.py`` makes of :mod:`fm.views`: an AST walk of every module under ``fm.web``
(imports inside functions included) and a fresh interpreter's ``sys.modules`` after importing the package. The read
views bring in ``fm.browser.session`` (the cookie reader, no write flow) through ``fm.espn.auth``; that one module is
tolerated, nothing else under ``fm.browser``, and nothing of the executor, the tick, the runner, the advisor, the MCP
stack or the CLI. The decisions the dashboard records are the human ones (approve, reject, pause, resume); it imports
no function that proposes, auto-approves, expires or executes.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import fm.web
import fm.web.app as web_app

WEB = Path(fm.web.__file__ or "").parent
FORBIDDEN_MODULES = (
    "fm.executor",
    "fm.browser",
    "playwright",
    "fm.jobs.tick",
    "fm.jobs.runner",
    "fm.advisor",
    "fm.mcp_server",
    "fastmcp",
    "mcp",
    "fm.commands",
    "fm.cli",
)
FORBIDDEN_NAMES = frozenset(
    {"propose", "begin_execution", "finish_execution", "auto_approve_due", "expire_due", "execute", "execute_approved"}
)


def _within(module: str, roots: tuple[str, ...]) -> bool:
    return any(module == root or module.startswith(f"{root}.") for root in roots)


def _imports(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):  # every import, including ones inside functions
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return modules, names


def test_no_module_under_fm_web_imports_the_executor_the_browser_or_a_write_function() -> None:
    sources = sorted(WEB.rglob("*.py"))
    assert {path.name for path in sources} >= {"__init__.py", "app.py", "server.py"}
    seen_modules: set[str] = set()
    for path in sources:
        modules, names = _imports(path)
        seen_modules |= modules
        assert {module for module in modules if _within(module, FORBIDDEN_MODULES)} == set(), path.name
        assert names & FORBIDDEN_NAMES == set(), path.name
    assert {"fm.views", "fm.proposals", "fm.config_edit", "uvicorn"} <= seen_modules  # the walk saw real imports


def test_importing_the_dashboard_loads_no_executor_tick_or_write_flow() -> None:
    code = (
        "import sys, fm.web.app, fm.web.server; "
        f"roots = {FORBIDDEN_MODULES!r}; "
        "allowed = {'fm.browser', 'fm.browser.session', 'playwright'}; "
        "print([n for n in sorted(sys.modules) if n not in allowed and not n.startswith('playwright.') "
        "and any(n == r or n.startswith(r + '.') for r in roots)])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


def test_the_package_itself_is_import_light() -> None:
    """``fm run`` and ``fm ui`` name the default port at CLI start-up; that must not load the web stack."""
    heavy = ("starlette", "uvicorn", "fm.views", "fm.web.app")
    code = f"import sys, fm.web; print([n for n in {heavy!r} if n in sys.modules])"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


def test_the_app_cannot_reach_a_write_function_through_its_namespace() -> None:
    namespace = vars(web_app)
    assert FORBIDDEN_NAMES & set(namespace) == set()
    assert not any("executor" in name.lower() or "browser" in name.lower() for name in namespace)
