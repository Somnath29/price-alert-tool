#!/usr/bin/env python3
"""
Price Alert Tool
Wissen Technology Campus Assessment

Evaluates price series against configured alert rules (level thresholds and
moving average crossovers). Emits sorted alerts when crossing conditions occur.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


class PriceAlertError(Exception):
    """Custom exception for domain and validation errors."""
    pass


@dataclass(frozen=True)
class PriceRecord:
    date: str
    instrument: str
    close: float


@dataclass(frozen=True)
class LevelRule:
    id: str
    instrument: str
    type: str  # 'level'
    direction: str  # 'above' or 'below'
    level: float


@dataclass(frozen=True)
class MACrossRule:
    id: str
    instrument: str
    type: str  # 'ma_cross'
    short: int
    long: int
    direction: str  # 'above' or 'below'


Rule = Union[LevelRule, MACrossRule]


@dataclass(frozen=True)
class Alert:
    date: str
    rule_id: str
    instrument: str
    rule_type: str
    details: str

    def format(self) -> str:
        return f"{self.date} {self.rule_id} {self.instrument} {self.rule_type} {self.details}"


# ---------------------------------------------------------------------------
# Loading and Validation
# ---------------------------------------------------------------------------

def _parse_and_normalize_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if not (1000 <= dt.year <= 9999):
            raise ValueError
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise PriceAlertError(f"Invalid date format '{date_str}', expected 'YYYY-MM-DD'")


def load_prices(filepath: Union[str, Path]) -> Dict[str, List[PriceRecord]]:
    """
    Loads price records from a CSV file.
    Validates headers, types, and duplicate entries.
    Returns records grouped by instrument and sorted chronologically.
    """
    path = Path(filepath)
    if not path.is_file():
        raise PriceAlertError(f"Prices file not found or not a file: '{filepath}'")

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
            if header_row is None:
                raise PriceAlertError(f"Prices file is empty: '{filepath}'")

            headers = [h.strip().lower() for h in header_row]
            required_cols = {"date", "instrument", "close"}
            if not required_cols.issubset(set(headers)):
                raise PriceAlertError(
                    f"Prices CSV header missing required columns. Found {headers}, expected 'date,instrument,close'"
                )

            date_idx = headers.index("date")
            inst_idx = headers.index("instrument")
            close_idx = headers.index("close")

            prices_by_instrument: Dict[str, List[PriceRecord]] = {}
            seen_entries = set()

            for line_no, row in enumerate(reader, start=2):
                if not row or all(c.strip() == "" for c in row):
                    continue  # skip blank lines

                if len(row) < len(headers):
                    raise PriceAlertError(
                        f"Malformed row at line {line_no} in '{filepath}': expected {len(headers)} columns, got {len(row)}"
                    )

                raw_date = row[date_idx].strip()
                inst_val = row[inst_idx].strip()
                close_raw = row[close_idx].strip()

                if not inst_val:
                    raise PriceAlertError(f"Empty instrument name at line {line_no} in '{filepath}'")

                date_val = _parse_and_normalize_date(raw_date)

                try:
                    close_val = float(close_raw)
                    if not math.isfinite(close_val) or close_val < 0:
                        raise ValueError
                except ValueError:
                    raise PriceAlertError(
                        f"Invalid close price '{close_raw}' at line {line_no} in '{filepath}': must be a finite, non-negative number"
                    )

                entry_key = (inst_val, date_val)
                if entry_key in seen_entries:
                    raise PriceAlertError(
                        f"Duplicate price entry for instrument '{inst_val}' on date '{date_val}' at line {line_no}"
                    )
                seen_entries.add(entry_key)

                if inst_val not in prices_by_instrument:
                    prices_by_instrument[inst_val] = []

                prices_by_instrument[inst_val].append(PriceRecord(date=date_val, instrument=inst_val, close=close_val))

    except OSError as e:
        raise PriceAlertError(f"Error reading prices file '{filepath}': {e}")

    # Sort each instrument's price records chronologically
    for inst in prices_by_instrument:
        prices_by_instrument[inst].sort(key=lambda r: r.date)

    return prices_by_instrument


def load_rules(filepath: Union[str, Path]) -> List[Rule]:
    """
    Loads alert rules from a JSON file and validates schema/parameters.
    """
    path = Path(filepath)
    if not path.is_file():
        raise PriceAlertError(f"Rules file not found or not a file: '{filepath}'")

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise PriceAlertError(f"Malformed JSON in rules file '{filepath}': {e}")
    except OSError as e:
        raise PriceAlertError(f"Error reading rules file '{filepath}': {e}")

    if not isinstance(data, list):
        raise PriceAlertError(f"Rules file must contain a JSON array of rule objects, got {type(data).__name__}")

    rules: List[Rule] = []
    seen_ids = set()

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise PriceAlertError(f"Rule at index {idx} must be a JSON object")

        rule_id = str(item.get("id", "")).strip()
        if not rule_id:
            raise PriceAlertError(f"Rule at index {idx} missing non-empty 'id'")

        if rule_id in seen_ids:
            raise PriceAlertError(f"Duplicate rule id '{rule_id}' found in rules file")
        seen_ids.add(rule_id)

        inst = str(item.get("instrument", "")).strip()
        if not inst:
            raise PriceAlertError(f"Rule '{rule_id}' missing non-empty 'instrument'")

        rule_type = str(item.get("type", "")).strip().lower()
        direction = str(item.get("direction", "")).strip().lower()

        if direction not in ("above", "below"):
            raise PriceAlertError(
                f"Rule '{rule_id}' has invalid direction '{direction}'. Must be 'above' or 'below'"
            )

        if rule_type == "level":
            if "level" not in item:
                raise PriceAlertError(f"Level rule '{rule_id}' missing 'level' parameter")
            level_val = item["level"]
            if isinstance(level_val, bool) or not isinstance(level_val, (int, float)):
                raise PriceAlertError(f"Level rule '{rule_id}' requires a numeric 'level', got {level_val!r}")
            if not math.isfinite(level_val) or level_val < 0:
                raise PriceAlertError(f"Level rule '{rule_id}' has invalid level {level_val}")

            rules.append(
                LevelRule(
                    id=rule_id,
                    instrument=inst,
                    type="level",
                    direction=direction,
                    level=float(level_val),
                )
            )

        elif rule_type == "ma_cross":
            if "short" not in item or "long" not in item:
                raise PriceAlertError(f"ma_cross rule '{rule_id}' requires both 'short' and 'long' window parameters")

            short_val = item["short"]
            long_val = item["long"]

            if isinstance(short_val, bool) or not isinstance(short_val, int) or short_val < 1:
                raise PriceAlertError(
                    f"ma_cross rule '{rule_id}' requires positive integer 'short' window, got {short_val!r}"
                )
            if isinstance(long_val, bool) or not isinstance(long_val, int) or long_val < 1:
                raise PriceAlertError(
                    f"ma_cross rule '{rule_id}' requires positive integer 'long' window, got {long_val!r}"
                )

            if short_val >= long_val:
                raise PriceAlertError(
                    f"ma_cross rule '{rule_id}' requires short window < long window (got short={short_val}, long={long_val})"
                )

            rules.append(
                MACrossRule(
                    id=rule_id,
                    instrument=inst,
                    type="ma_cross",
                    short=short_val,
                    long=long_val,
                    direction=direction,
                )
            )

        else:
            raise PriceAlertError(f"Rule '{rule_id}' has unsupported type '{rule_type}'. Supported: level, ma_cross")

    return rules


# ---------------------------------------------------------------------------
# Rule Evaluation
# ---------------------------------------------------------------------------

def _format_level(val: float) -> str:
    """Formats level numbers cleanly (e.g. 1500 instead of 1500.0 if whole)."""
    return str(int(val)) if val.is_integer() else f"{val:g}"


def evaluate_level_rule(rule: LevelRule, records: List[PriceRecord]) -> List[Alert]:
    """
    Evaluates a level rule on a chronologically sorted list of price records.
    A level rule fires on the first day the close crosses the level in the given
    direction (it was on the other side the previous day).
    Sustained conditions on subsequent days do not fire.
    """
    alerts: List[Alert] = []
    if len(records) < 2:
        return alerts

    # Track prior side: +1 if above, -1 if below, None if initial/unassigned
    # When price is exactly at level, side is 0 (boundary)
    prev_side: Optional[int] = None

    first_close = records[0].close
    if first_close > rule.level:
        prev_side = 1
    elif first_close < rule.level:
        prev_side = -1
    else:
        prev_side = 0

    formatted_level = _format_level(rule.level)

    for r in records[1:]:
        curr_close = r.close
        if curr_close > rule.level:
            curr_side = 1
        elif curr_close < rule.level:
            curr_side = -1
        else:
            curr_side = 0

        if rule.direction == "below":
            # Crossed below: currently below, and previously on the other side (above)
            if curr_side == -1 and prev_side == 1:
                alerts.append(
                    Alert(
                        date=r.date,
                        rule_id=rule.id,
                        instrument=rule.instrument,
                        rule_type="level",
                        details=f"below {formatted_level}",
                    )
                )
        elif rule.direction == "above":
            # Crossed above: currently above, and previously on the other side (below)
            if curr_side == 1 and prev_side == -1:
                alerts.append(
                    Alert(
                        date=r.date,
                        rule_id=rule.id,
                        instrument=rule.instrument,
                        rule_type="level",
                        details=f"above {formatted_level}",
                    )
                )

        # Update previous side when price moves off boundary
        if curr_side != 0:
            prev_side = curr_side

    return alerts


def evaluate_ma_cross_rule(rule: MACrossRule, records: List[PriceRecord]) -> List[Alert]:
    """
    Evaluates an ma_cross rule on a chronologically sorted list of price records.
    Requires full windows of closing prices.
    An ma_cross rule fires on the day the short moving average crosses the long one
    in the given direction.
    """
    alerts: List[Alert] = []
    short_w = rule.short
    long_w = rule.long

    # Need at least long_w observations to compute the first long moving average
    if len(records) < long_w:
        return alerts

    closes = [r.close for r in records]

    def get_ma(idx: int, window: int) -> float:
        # Arithmetic mean of closing prices over the full window
        return sum(closes[idx - window + 1 : idx + 1]) / window

    prev_side: Optional[int] = None

    # First index where both MAs are defined is long_w - 1
    init_idx = long_w - 1
    short_ma_init = get_ma(init_idx, short_w)
    long_ma_init = get_ma(init_idx, long_w)

    if short_ma_init > long_ma_init:
        prev_side = 1
    elif short_ma_init < long_ma_init:
        prev_side = -1
    else:
        prev_side = 0

    details = f"{short_w}/{long_w} {rule.direction}"

    for i in range(long_w, len(records)):
        r = records[i]
        short_ma = get_ma(i, short_w)
        long_ma = get_ma(i, long_w)

        if short_ma > long_ma:
            curr_side = 1
        elif short_ma < long_ma:
            curr_side = -1
        else:
            curr_side = 0

        if rule.direction == "above":
            # Crossed above: short was <= long (specifically on the below side) and now strictly above
            if curr_side == 1 and prev_side == -1:
                alerts.append(
                    Alert(
                        date=r.date,
                        rule_id=rule.id,
                        instrument=rule.instrument,
                        rule_type="ma_cross",
                        details=details,
                    )
                )
        elif rule.direction == "below":
            # Crossed below: short was >= long (specifically on the above side) and now strictly below
            if curr_side == -1 and prev_side == 1:
                alerts.append(
                    Alert(
                        date=r.date,
                        rule_id=rule.id,
                        instrument=rule.instrument,
                        rule_type="ma_cross",
                        details=details,
                    )
                )

        if curr_side != 0:
            prev_side = curr_side

    return alerts


def evaluate_rules(
    prices_by_instrument: Dict[str, List[PriceRecord]],
    rules: List[Rule],
) -> List[Alert]:
    """
    Evaluates all rules against the loaded prices.
    Returns alerts sorted chronologically by date, then by rule ID.
    """
    all_alerts: List[Alert] = []

    for rule in rules:
        records = prices_by_instrument.get(rule.instrument, [])
        if not records:
            continue

        if isinstance(rule, LevelRule):
            rule_alerts = evaluate_level_rule(rule, records)
        elif isinstance(rule, MACrossRule):
            rule_alerts = evaluate_ma_cross_rule(rule, records)
        else:
            continue

        all_alerts.extend(rule_alerts)

    # Sort alerts by date ascending, then rule ID ascending
    all_alerts.sort(key=lambda a: (a.date, a.rule_id))
    return all_alerts


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def run_cli(argv: Optional[List[str]] = None) -> int:
    """
    CLI runner. Parses arguments, evaluates alerts, prints results.
    Returns exit code (0 for success, non-zero for error).
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        sys.stderr.write("Usage: python alerts.py <prices.csv> <rules.json>\n")
        return 1

    prices_path, rules_path = argv

    try:
        prices_by_instrument = load_prices(prices_path)
        rules = load_rules(rules_path)
        alerts = evaluate_rules(prices_by_instrument, rules)

        for alert in alerts:
            print(alert.format())

        return 0

    except PriceAlertError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
