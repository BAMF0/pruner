"""Record real Launchpad payloads as test fixtures.

Run manually when fixtures need refreshing:

    uv run python -m tests.record_fixtures

Tests never hit the network; they replay what this script captured. Using real
payloads rather than hand-written dicts is deliberate -- the parsing code exists
precisely to cope with Launchpad's actual quirks (``latest_patch_uploaded`` being
null on a bug that has a Patch attachment, message 0 being the description, and
so on), and invented fixtures would not exercise any of that.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

ROOT = "https://api.launchpad.net/devel"
OUT = Path(__file__).parent / "fixtures" / "lp"

#: Chosen to cover the interesting shapes:
#:   1374898 -- apport metadata, series tag (trusty), Patch attachment, comments
#:   717691  -- ancient, legacy "Binary package hint:" preamble, no apport block
#:   1509299 -- High importance (exercises the protected-importance exclusion)
BUGS = (1374898, 717691, 1509299)

SUB_COLLECTIONS = (
    "bug_tasks",
    "messages",
    "attachments",
    "bug_watches",
    "cves",
    "vulnerabilities",
    "linked_merge_proposals",
    "linked_branches",
)


def save(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(Path.cwd())}")


def main() -> None:
    with httpx.Client(timeout=60.0, headers={"Accept": "application/json"}) as client:

        def get(url: str, **params: str) -> object:
            response = client.get(url, params=params or None)
            response.raise_for_status()
            return response.json()

        save("ubuntu_series", get(f"{ROOT}/ubuntu/series", **{"ws.size": "75"}))

        for bug_id in BUGS:
            save(f"bug_{bug_id}", get(f"{ROOT}/bugs/{bug_id}"))
            for collection in SUB_COLLECTIONS:
                save(
                    f"bug_{bug_id}_{collection}",
                    get(f"{ROOT}/bugs/{bug_id}/{collection}", **{"ws.size": "75"}),
                )

        save(
            "vim_published_noble",
            get(
                f"{ROOT}/ubuntu/+archive/primary",
                **{
                    "ws.op": "getPublishedSources",
                    "source_name": "vim",
                    "exact_match": "true",
                    "status": "Published",
                    "distro_series": f"{ROOT}/ubuntu/noble",
                    "ws.size": "20",
                },
            ),
        )


if __name__ == "__main__":
    main()
