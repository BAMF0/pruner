"""Parsing of apport metadata out of Launchpad bug descriptions.

``ubuntu-bug`` appends a machine-readable block to the description, which is often
the *only* record of which release a bug was filed against::

    ProblemType: Bug
    DistroRelease: Ubuntu 14.04
    Package: vim-gtk 2:7.4.052-1ubuntu3
    Uname: Linux 3.13.0-35-generic x86_64
    ...

Two jobs here:

1. Extract ``DistroRelease`` / ``Package`` so the EOL and likely-fixed rules have
   something concrete to reason about.
2. Strip the block, so "how much did the human actually write?" is measured on prose
   rather than on kilobytes of boilerplate. This matters for the ``empty_report``
   rule and it keeps LLM prompts small.

Everything here is best-effort: a description with no recognisable metadata simply
yields an empty :class:`~pruner.models.ApportInfo`, and the dependent rules then do
not fire. Absence of evidence never produces an action.
"""

from __future__ import annotations

import re

from pruner.models import ApportInfo

#: ``Key: value`` at the start of a line. Apport keys are CamelCase.
_KEY_RE = re.compile(r"^(?P<key>[A-Z][A-Za-z0-9_.+-]*):(?:[ \t]+(?P<value>.*))?$")

#: Keys distinctive enough to positively identify an apport block. We only strip
#: when one of these is present, so a bug that happens to contain a line like
#: "Steps: ..." is not mistaken for metadata.
_ANCHOR_KEYS = frozenset(
    {
        "ProblemType",
        "DistroRelease",
        "ApportVersion",
        "ProcVersionSignature",
        "Uname",
        "NonfreeKernelModules",
        "SourcePackage",
        "InstallationDate",
        "InstallationMedia",
        "UpgradeStatus",
        "Dependencies",
    }
)

#: Legacy Malone/apport preamble seen on older reports, e.g. bug #717691.
_BINARY_HINT_RE = re.compile(r"^Binary package hint:.*$", re.MULTILINE)

#: e.g. "Ubuntu 14.04", "Ubuntu 14.04.1 LTS" -> 14.04
_RELEASE_RE = re.compile(r"(?P<release>\d{1,2}\.\d{2})")

#: Trailing annotations apport adds to a version, e.g. "1.2-3 [modified: foo]".
_VERSION_NOISE_RE = re.compile(r"[\[(].*$")


def _iter_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        if match := _KEY_RE.match(line):
            pairs.append((match["key"], (match["value"] or "").strip()))
    return pairs


def parse_apport(description: str) -> ApportInfo:
    """Recover structured apport fields from a bug description."""
    if not description:
        return ApportInfo()

    fields = dict(_iter_pairs(description))

    distro_release: str | None = None
    raw_release = fields.get("DistroRelease")
    if raw_release and (match := _RELEASE_RE.search(raw_release)):
        distro_release = match["release"]

    package: str | None = None
    version: str | None = None
    if raw := fields.get("Package"):
        parts = raw.split()
        if parts:
            package = parts[0]
        if len(parts) > 1:
            candidate = _VERSION_NOISE_RE.sub("", parts[1]).strip()
            # Apport writes "(not installed)" when it cannot determine a version.
            if candidate and any(ch.isdigit() for ch in candidate):
                version = candidate

    return ApportInfo(
        distro_release=distro_release,
        package=package,
        version=version,
        problem_type=fields.get("ProblemType") or None,
    )


def strip_apport(description: str) -> str:
    """Return the description with the trailing apport block removed.

    The block is anchored on a distinctive key and then extended backwards over any
    contiguous ``Key: value`` lines, so a ``Package:`` line preceding the anchor is
    also removed. If no anchor is found the text is returned essentially unchanged.
    """
    if not description:
        return ""

    text = _BINARY_HINT_RE.sub("", description)
    lines = text.splitlines()

    anchor: int | None = None
    for index, line in enumerate(lines):
        match = _KEY_RE.match(line)
        if match and match["key"] in _ANCHOR_KEYS:
            anchor = index
            break

    if anchor is None:
        return text.strip()

    # Walk backwards over contiguous metadata-looking lines to find the true start.
    start = anchor
    while start > 0:
        previous = lines[start - 1]
        if _KEY_RE.match(previous) or not previous.strip():
            start -= 1
        else:
            break

    return "\n".join(lines[:start]).strip()


def prose_length(description: str) -> int:
    """Length of the human-written portion of a description, in characters.

    Used by the ``empty_report`` rule. Whitespace is collapsed so that a description
    padded with blank lines does not look substantial.
    """
    return len(re.sub(r"\s+", " ", strip_apport(description)).strip())
