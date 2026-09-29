"""Launchpad access.

Split deliberately by capability:

* :mod:`pruner.lp.read` -- anonymous HTTP GETs, no credentials, cannot write.
* :mod:`pruner.lp.series` -- series status, derived from read-only data.
* :mod:`pruner.lp.archive` -- source publication lookups, read-only.
* :mod:`pruner.lp.write` -- the *only* module that authenticates and mutates.

Keeping the OAuth import isolated to ``write`` means a bug anywhere in the
fetch/analyze/report path cannot possibly modify a bug report.
"""
