"""Shared test fixtures.

The series table is pinned to a frozen copy of the recorded payload *with the
status fields overridden to known values*, and "now" is pinned too. Without both,
the whole suite would start failing the moment Ubuntu releases a version -- the
EOL rules are functions of the calendar, so the calendar has to be an input.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pruner.config import Config
from pruner.lp.archive import ArchiveIndex, Publication
from pruner.lp.series import SeriesTable
from pruner.models import ApportInfo, BugSnapshot, BugTaskStatus, Importance, TaskSnapshot

FIXTURES = Path(__file__).parent / "fixtures" / "lp"

#: Pinned "current time" for all tests.
NOW = datetime(2026, 9, 28, tzinfo=UTC)


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


#: A deterministic series table. Deliberately not the live one: these statuses are
#: the test's premise, not something to be discovered at runtime.
SERIES_ENTRIES: list[dict[str, Any]] = [
    {
        "name": "stonking",
        "version": "26.10",
        "status": "Active Development",
        "supported": False,
        "active": True,
        "self_link": "https://api.launchpad.net/devel/ubuntu/stonking",
    },
    {
        "name": "resolute",
        "version": "26.04",
        "status": "Current Stable Release",
        "supported": True,
        "active": True,
        "self_link": "https://api.launchpad.net/devel/ubuntu/resolute",
    },
    {
        "name": "noble",
        "version": "24.04",
        "status": "Supported",
        "supported": True,
        "active": True,
        "self_link": "https://api.launchpad.net/devel/ubuntu/noble",
    },
    {
        "name": "jammy",
        "version": "22.04",
        "status": "Supported",
        "supported": True,
        "active": True,
        "self_link": "https://api.launchpad.net/devel/ubuntu/jammy",
    },
    {
        "name": "questing",
        "version": "25.10",
        "status": "Obsolete",
        "supported": False,
        "active": False,
        "self_link": "https://api.launchpad.net/devel/ubuntu/questing",
    },
    {
        "name": "focal",
        "version": "20.04",
        "status": "Obsolete",
        "supported": False,
        "active": False,
        "self_link": "https://api.launchpad.net/devel/ubuntu/focal",
    },
    {
        "name": "trusty",
        "version": "14.04",
        "status": "Obsolete",
        "supported": False,
        "active": False,
        "self_link": "https://api.launchpad.net/devel/ubuntu/trusty",
    },
    {
        "name": "lucid",
        "version": "10.04",
        "status": "Obsolete",
        "supported": False,
        "active": False,
        "self_link": "https://api.launchpad.net/devel/ubuntu/lucid",
    },
]


@pytest.fixture
def series() -> SeriesTable:
    return SeriesTable.from_api("ubuntu", SERIES_ENTRIES)


@pytest.fixture
def config() -> Config:
    """Built-in defaults, independent of any pruner.toml in the working tree.

    ``action_delay_seconds`` is zeroed: the delay exists to be polite to the
    Launchpad API, and there is no API here. Tests that care about the delay
    should set it explicitly.
    """
    return Config.model_validate({"safety": {"action_delay_seconds": 0.0}})


@pytest.fixture
def archive() -> ArchiveIndex:
    return ArchiveIndex(
        distribution="ubuntu",
        package="vim",
        publications=(
            Publication(series="jammy", version="2:8.2.3995-1ubuntu2.24", pocket="Updates"),
            Publication(series="noble", version="2:9.1.0016-1ubuntu7.20", pocket="Updates"),
            Publication(series="resolute", version="2:9.1.1500-1ubuntu1", pocket="Release"),
            Publication(series="stonking", version="2:9.1.1600-1ubuntu1", pocket="Release"),
        ),
        queried_series=("stonking", "resolute", "noble", "jammy"),
    )


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def make_task(
    *,
    target: str = "vim (Ubuntu)",
    status: BugTaskStatus = BugTaskStatus.NEW,
    importance: Importance = Importance.UNDECIDED,
    series_name: str | None = None,
    package: str | None = "vim",
    assignee: str | None = None,
    milestone: str | None = None,
    link: str | None = None,
) -> TaskSnapshot:
    return TaskSnapshot(
        target_name=target,
        package=package,
        distribution="ubuntu",
        series=series_name,
        status=status,
        importance=importance,
        assignee=assignee,
        milestone=milestone,
        is_complete=False,
        self_link=link or f"https://api.launchpad.net/devel/task/{target.replace(' ', '_')}",
    )


def make_bug(
    *,
    bug_id: int = 1,
    title: str = "vim crashes when opening a file",
    description: str = "x" * 400,
    tags: tuple[str, ...] = (),
    tasks: tuple[TaskSnapshot, ...] | None = None,
    apport: ApportInfo | None = None,
    comment_texts: tuple[str, ...] = (),
    quiet_days: int = 3000,
    enriched: bool = True,
    **overrides: Any,
) -> BugSnapshot:
    """A bug that is, by default, boring enough to be prunable.

    Tests then set exactly the one field they are about, which keeps each test's
    premise obvious.
    """
    last = NOW.timestamp() - quiet_days * 86400
    stamp = datetime.fromtimestamp(last, tz=UTC)
    fields: dict[str, Any] = {
        "id": bug_id,
        "title": title,
        "description": description,
        "web_link": f"https://bugs.launchpad.net/bugs/{bug_id}",
        "tags": tags,
        "tasks": tasks if tasks is not None else (make_task(),),
        "apport": apport or ApportInfo(),
        "comment_texts": comment_texts,
        "date_created": stamp,
        "date_last_updated": stamp,
        "date_last_message": stamp,
        "message_count": 1,
        "enriched": enriched,
    }
    fields.update(overrides)
    return BugSnapshot(**fields)
