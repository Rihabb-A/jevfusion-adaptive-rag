"""
Team-wide API spend ledger (course cap: USD 20 total for the project).

Every paid call batch appends one row to results/budget_ledger.csv with
the provider-reported cost. Scripts call `ensure_affordable(estimate)`
before a run, which stops if the ledger total plus the estimate would
exceed BUDGET_CAP_USD from .env.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "budget_ledger.csv"
FIELDS = ["timestamp_utc", "component", "model", "calls", "input_tokens", "cost_usd", "note"]


def spent():
    if not LEDGER.exists():
        return 0.0
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        return sum(float(row["cost_usd"]) for row in csv.DictReader(handle))


def ensure_affordable(estimate_usd):
    cap = float(os.getenv("BUDGET_CAP_USD", "20"))
    total = spent()
    if total + estimate_usd > cap:
        raise SystemExit(
            f"Refusing to run: ledger ${total:.4f} + estimate ${estimate_usd:.4f} "
            f"exceeds BUDGET_CAP_USD ${cap:.2f}"
        )
    print(f"Budget: spent ${total:.4f}, this run est. ${estimate_usd:.4f}, cap ${cap:.2f}")


def record(component, model, calls, input_tokens, cost_usd, note=""):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LEDGER.exists()
    with LEDGER.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "component": component,
            "model": model,
            "calls": calls,
            "input_tokens": input_tokens,
            "cost_usd": f"{cost_usd:.8f}",
            "note": note,
        })
