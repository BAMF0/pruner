"""Standard-support end dates from ``distro-info``.

Necessary because Launchpad's series ``status`` does not mean what a bug triager
needs it to mean. Verified against live data: Launchpad reports trusty (14.04),
xenial (16.04), bionic (18.04) and focal (20.04) all as ``Supported`` with
``supported: true``, because they remain covered by Ubuntu Pro / ESM. Launchpad
exposes ``datereleased`` but no end-of-life date at all.

For triage purposes that is the wrong notion. ESM provides security updates for a
subset of packages to subscribers; it does not mean anyone is going to fix a
general functional bug on a twelve-year-old release. Treating ESM releases as
supported would make the EOL rules fire on essentially nothing.

``/usr/share/distro-info/ubuntu.csv`` (from the ``distro-info-data`` package, on
every Ubuntu system) carries the distinction explicitly::

    version,codename,series,created,release,eol,eol-server,eol-esm,eol-legacy
    14.04 LTS,Trusty Tahr,trusty,2013-10-17,2014-04-17,2019-04-25,...,2024-04-25,...

``eol`` is the end of standard support; ``eol-esm`` is the extended one. This
module reads ``eol``, which is the same column ``ubuntu-distro-info --supported``
uses.

If the file is absent (a non-Ubuntu host, or a container without
``distro-info-data``), lookups return ``None`` and the caller falls back to
trusting Launchpad -- which errs towards treating releases as supported, and so
towards pruning less.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

def csv_path_for(distribution: str) -> Path:
    return Path(f"/usr/share/distro-info/{distribution.lower()}.csv")


def load_eol_dates(distribution: str, *, path: Path | None = None) -> dict[str, date]:
    """Map series codename -> end of *standard* support.

    Returns an empty mapping if the data is unavailable, which the caller must
    treat as "no information" rather than as "nothing is EOL".
    """
    resolved = path or csv_path_for(distribution)
    if not resolved.is_file():
        log.info(
            "no distro-info data at %s; falling back to Launchpad series status "
            "(ESM releases will count as supported)",
            resolved,
        )
        return {}

    dates: dict[str, date] = {}
    try:
        with resolved.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                series = (row.get("series") or "").strip().lower()
                raw = (row.get("eol") or "").strip()
                if not series or not raw:
                    continue
                try:
                    dates[series] = date.fromisoformat(raw)
                except ValueError:
                    continue
    except OSError:
        log.warning("could not read %s; falling back to Launchpad status", resolved)
        return {}

    return dates
