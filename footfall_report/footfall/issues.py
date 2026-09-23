"""Data-quality findings, kept as plain objects so any front-end can render them."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
SEVERITY_ICON = {"error": "Error", "warning": "Warning", "info": "Info"}


@dataclass
class Issue:
    """One data-quality finding.

    severity: "error"   -> the report cannot be trusted / cannot be produced
              "warning" -> rows were rejected or look wrong, totals exclude them
              "info"    -> worth knowing, nothing was dropped
    """

    severity: str
    category: str
    message: str
    rows: list[int] = field(default_factory=list)

    def rows_label(self, limit: int = 25) -> str:
        if not self.rows:
            return ""
        shown = ", ".join(str(r) for r in self.rows[:limit])
        if len(self.rows) > limit:
            shown += f", ... (+{len(self.rows) - limit} more)"
        return shown


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    """Render issues as a table for display/export."""
    rows = [
        {
            "Severity": SEVERITY_ICON.get(i.severity, i.severity),
            "Category": i.category,
            "Finding": i.message,
            "CSV rows": i.rows_label(),
        }
        for i in sorted(issues, key=lambda i: (SEVERITY_ORDER.get(i.severity, 9), i.category))
    ]
    return pd.DataFrame(rows, columns=["Severity", "Category", "Finding", "CSV rows"])
