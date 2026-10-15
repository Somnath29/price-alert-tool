#!/usr/bin/env python3
"""
Unit and Integration Tests for Price Alert Tool
Tests all required cases:
- Both rule types (level, ma_cross)
- Crossings versus sustained conditions
- Unsorted CSV dates
- Insufficient MA history
- Multiple instruments evaluated independently
- Output sorting (by date ascending, then rule ID)
- No alerts fired
- Invalid input files and rule parameters
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from alerts import (
    Alert,
    LevelRule,
    MACrossRule,
    PriceAlertError,
    PriceRecord,
    evaluate_level_rule,
    evaluate_ma_cross_rule,
    evaluate_rules,
    load_prices,
    load_rules,
    run_cli,
)


class TestLevelRule(unittest.TestCase):
    """Tests for level threshold crossing detection."""

    def test_level_cross_below(self):
        rule = LevelRule(id="r1", instrument="INFY", type="level", direction="below", level=1500.0)
        records = [
            PriceRecord(date="2026-09-01", instrument="INFY", close=1520.0),
            PriceRecord(date="2026-09-02", instrument="INFY", close=1510.0),
            PriceRecord(date="2026-09-03", instrument="INFY", close=1490.0),  # Crossing below
        ]
        alerts = evaluate_level_rule(rule, records)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].date, "2026-09-03")
        self.assertEqual(alerts[0].format(), "2026-09-03 r1 INFY level below 1500")

    def test_level_cross_above(self):
        rule = LevelRule(id="r_above", instrument="AAPL", type="level", direction="above", level=200.0)
        records = [
            PriceRecord(date="2026-01-01", instrument="AAPL", close=190.0),
            PriceRecord(date="2026-01-02", instrument="AAPL", close=195.0),
            PriceRecord(date="2026-01-03", instrument="AAPL", close=205.0),  # Crossing above
        ]
        alerts = evaluate_level_rule(rule, records)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].date, "2026-01-03")
        self.assertEqual(alerts[0].format(), "2026-01-03 r_above AAPL level above 200")

    def test_level_sustained_condition_does_not_retrigger(self):
        """Alerts must fire only on the transition day, not during sustained states."""
        rule = LevelRule(id="r1", instrument="INFY", type="level", direction="below", level=1500.0)
        records = [
            PriceRecord(date="2026-09-01", instrument="INFY", close=1520.0),
            PriceRecord(date="2026-09-02", instrument="INFY", close=1490.0),  # Crossing below -> Alert 1
            PriceRecord(date="2026-09-03", instrument="INFY", close=1480.0),  # Sustained below -> No alert
            PriceRecord(date="2026-09-04", instrument="INFY", close=1470.0),  # Sustained below -> No alert
            PriceRecord(date="2026-09-05", instrument="INFY", close=1510.0),  # Back above -> No alert
            PriceRecord(date="2026-09-06", instrument="INFY", close=1495.0),  # Crossing below again -> Alert 2
        ]
        alerts = evaluate_level_rule(rule, records)
        self.assertEqual(len(alerts), 2)
        self.assertEqual([a.date for a in alerts], ["2026-09-02", "2026-09-06"])

    def test_level_starting_below_and_staying_below_no_alert(self):
        """If price starts below the level and never was above, no crossing has occurred."""
        rule = LevelRule(id="r1", instrument="INFY", type="level", direction="below", level=1500.0)
        records = [
            PriceRecord(date="2026-09-01", instrument="INFY", close=1480.0),
            PriceRecord(date="2026-09-02", instrument="INFY", close=1470.0),
        ]
        alerts = evaluate_level_rule(rule, records)
        self.assertEqual(len(alerts), 0)

    def test_level_exact_boundary_touch(self):
        """Touching the exact level boundary should not cause false crossing alerts."""
        rule = LevelRule(id="r1", instrument="INFY", type="level", direction="below", level=1500.0)
        records = [
            PriceRecord(date="2026-09-01", instrument="INFY", close=1520.0),
            PriceRecord(date="2026-09-02", instrument="INFY", close=1500.0),  # Exactly at level
            PriceRecord(date="2026-09-03", instrument="INFY", close=1490.0),  # Crosses below
        ]
        alerts = evaluate_level_rule(rule, records)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].date, "2026-09-03")

    def test_level_exact_boundary_bounce_no_alert(self):
        """
        Price rising to touch a resistance level exactly and falling back down
        must not trigger an 'above' alert (bounce, not a crossing).
        """
        rule = LevelRule(id="r_bounce", instrument="INFY", type="level", direction="above", level=1500.0)
        records = [
            PriceRecord(date="2026-09-01", instrument="INFY", close=1480.0),  # Below
            PriceRecord(date="2026-09-02", instrument="INFY", close=1500.0),  # Touching resistance
            PriceRecord(date="2026-09-03", instrument="INFY", close=1490.0),  # Rejected back down
        ]
        alerts = evaluate_level_rule(rule, records)
        self.assertEqual(len(alerts), 0)

    def test_level_cross_above_via_exact_boundary(self):
        """
        Price rising from below, pausing on the boundary, and continuing above
        triggers an alert on the breakout day (day 3), not on the boundary day.
        """
        rule = LevelRule(id="r_breakout", instrument="INFY", type="level", direction="above", level=1500.0)
        records = [
            PriceRecord(date="2026-09-01", instrument="INFY", close=1480.0),  # Below
            PriceRecord(date="2026-09-02", instrument="INFY", close=1500.0),  # Boundary touch
            PriceRecord(date="2026-09-03", instrument="INFY", close=1520.0),  # Breakout above
        ]
        alerts = evaluate_level_rule(rule, records)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].date, "2026-09-03")
        self.assertEqual(alerts[0].format(), "2026-09-03 r_breakout INFY level above 1500")



class TestMACrossRule(unittest.TestCase):
    """Tests for moving average crossover detection."""

    def test_ma_cross_above(self):
        rule = MACrossRule(id="r_ma", instrument="TCS", type="ma_cross", short=2, long=4, direction="above")
        # short=2, long=4
        # Day 1: 10
        # Day 2: 10
        # Day 3: 10
        # Day 4: 10 -> MA2=10, MA4=10 (initial defined day, prev_side=0)
        # Day 5: 8  -> MA2=9, MA4=9.5 -> short < long (prev_side=-1)
        # Day 6: 15 -> MA2=11.5, MA4=10.75 -> short > long (crosses above!)
        records = [
            PriceRecord(date="2026-01-01", instrument="TCS", close=10.0),
            PriceRecord(date="2026-01-02", instrument="TCS", close=10.0),
            PriceRecord(date="2026-01-03", instrument="TCS", close=10.0),
            PriceRecord(date="2026-01-04", instrument="TCS", close=10.0),
            PriceRecord(date="2026-01-05", instrument="TCS", close=8.0),
            PriceRecord(date="2026-01-06", instrument="TCS", close=15.0),
        ]
        alerts = evaluate_ma_cross_rule(rule, records)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].date, "2026-01-06")
        self.assertEqual(alerts[0].format(), "2026-01-06 r_ma TCS ma_cross 2/4 above")

    def test_ma_cross_below(self):
        rule = MACrossRule(id="r_below", instrument="TCS", type="ma_cross", short=2, long=3, direction="below")
        # Day 1: 20
        # Day 2: 20
        # Day 3: 30 -> MA2=25, MA3=23.33 (short > long, prev_side=+1)
        # Day 4: 5  -> MA2=17.5, MA3=18.33 (short < long, crosses below!)
        records = [
            PriceRecord(date="2026-01-01", instrument="TCS", close=20.0),
            PriceRecord(date="2026-01-02", instrument="TCS", close=20.0),
            PriceRecord(date="2026-01-03", instrument="TCS", close=30.0),
            PriceRecord(date="2026-01-04", instrument="TCS", close=5.0),
        ]
        alerts = evaluate_ma_cross_rule(rule, records)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].date, "2026-01-04")
        self.assertEqual(alerts[0].format(), "2026-01-04 r_below TCS ma_cross 2/3 below")

    def test_ma_cross_sustained_condition_does_not_retrigger(self):
        rule = MACrossRule(id="r_ma", instrument="TCS", type="ma_cross", short=2, long=3, direction="above")
        records = [
            PriceRecord(date="2026-01-01", instrument="TCS", close=30.0),
            PriceRecord(date="2026-01-02", instrument="TCS", close=10.0),
            PriceRecord(date="2026-01-03", instrument="TCS", close=10.0),  # MA2=10, MA3=16.67 (short < long)
            PriceRecord(date="2026-01-04", instrument="TCS", close=40.0),  # MA2=25, MA3=20.0 (crosses above -> Alert)
            PriceRecord(date="2026-01-05", instrument="TCS", close=50.0),  # MA2=45, MA3=33.3 (sustained -> No alert)
            PriceRecord(date="2026-01-06", instrument="TCS", close=60.0),  # MA2=55, MA3=50.0 (sustained -> No alert)
        ]
        alerts = evaluate_ma_cross_rule(rule, records)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].date, "2026-01-04")

    def test_ma_cross_insufficient_history(self):
        """When total observations are fewer than long window, no alerts fire."""
        rule = MACrossRule(id="r_ma", instrument="TCS", type="ma_cross", short=5, long=20, direction="above")
        records = [
            PriceRecord(date=f"2026-01-{i:02d}", instrument="TCS", close=100.0 + i)
            for i in range(1, 15)  # only 14 days, long window requires 20
        ]
        alerts = evaluate_ma_cross_rule(rule, records)
        self.assertEqual(len(alerts), 0)

    def test_ma_cross_earliest_possible_day(self):
        """
        For short=2, long=3, day 3 (observation count = long window) is the first
        day both MAs are defined and establishes the baseline relationship.
        The earliest day a crossover can trigger is day 4 (observation count = long + 1).
        """
        rule = MACrossRule(id="r_early", instrument="TCS", type="ma_cross", short=2, long=3, direction="above")
        # Day 1: 100, Day 2: 40, Day 3: 40 -> MA2=40, MA3=60 (short < long established, 0 alerts)
        records_3 = [
            PriceRecord(date="2026-01-01", instrument="TCS", close=100.0),
            PriceRecord(date="2026-01-02", instrument="TCS", close=40.0),
            PriceRecord(date="2026-01-03", instrument="TCS", close=40.0),
        ]
        alerts_day3 = evaluate_ma_cross_rule(rule, records_3)
        self.assertEqual(len(alerts_day3), 0)

        # Day 4: 100 -> MA2=70, MA3=60 (short > long -> Crossover fires on day 4)
        records_4 = records_3 + [PriceRecord(date="2026-01-04", instrument="TCS", close=100.0)]
        alerts_day4 = evaluate_ma_cross_rule(rule, records_4)
        self.assertEqual(len(alerts_day4), 1)
        self.assertEqual(alerts_day4[0].date, "2026-01-04")
        self.assertEqual(alerts_day4[0].format(), "2026-01-04 r_early TCS ma_cross 2/3 above")

    def test_ma_cross_repeated_oscillations(self):
        """
        In volatile markets, MAs may cross back and forth repeatedly.
        An 'above' rule must fire on every distinct upward crossing,
        while completely ignoring downward crosses and sustained above/below days.
        """
        rule = MACrossRule(id="r_osc", instrument="TCS", type="ma_cross", short=2, long=3, direction="above")
        records = [
            PriceRecord("2026-01-01", "TCS", 100.0),
            PriceRecord("2026-01-02", "TCS", 40.0),
            PriceRecord("2026-01-03", "TCS", 40.0),   # Baseline: MA2=40, MA3=60 (short < long)
            PriceRecord("2026-01-04", "TCS", 100.0),  # MA2=70, MA3=60 -> Cross above (Alert 1)
            PriceRecord("2026-01-05", "TCS", 100.0),  # MA2=100, MA3=80 -> Sustained above (no alert)
            PriceRecord("2026-01-06", "TCS", 20.0),   # MA2=60, MA3=73.33 -> Cross below (no alert for 'above')
            PriceRecord("2026-01-07", "TCS", 20.0),   # MA2=20, MA3=46.67 -> Sustained below (no alert)
            PriceRecord("2026-01-08", "TCS", 80.0),   # MA2=50, MA3=40 -> Cross above again (Alert 2)
        ]
        alerts = evaluate_ma_cross_rule(rule, records)
        self.assertEqual(len(alerts), 2)
        self.assertEqual([a.date for a in alerts], ["2026-01-04", "2026-01-08"])
        self.assertEqual(alerts[0].format(), "2026-01-04 r_osc TCS ma_cross 2/3 above")
        self.assertEqual(alerts[1].format(), "2026-01-08 r_osc TCS ma_cross 2/3 above")



class TestMultipleInstrumentsAndUnsortedDates(unittest.TestCase):
    """Tests multi-instrument independence, unordered dates, and output sorting."""

    def test_unordered_dates_are_sorted_per_instrument(self):
        csv_content = (
            "date,instrument,close\n"
            "2026-09-03,INFY,1490.00\n"
            "2026-09-01,INFY,1520.00\n"
            "2026-09-02,INFY,1510.00\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tf:
            tf.write(csv_content)
            tf_name = tf.name

        try:
            prices = load_prices(tf_name)
            self.assertEqual(
                [r.date for r in prices["INFY"]],
                ["2026-09-01", "2026-09-02", "2026-09-03"]
            )
            rule = LevelRule(id="r1", instrument="INFY", type="level", direction="below", level=1500.0)
            alerts = evaluate_rules(prices, [rule])
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].date, "2026-09-03")
        finally:
            os.remove(tf_name)

    def test_multiple_instruments_evaluated_independently(self):
        prices = {
            "INFY": [
                PriceRecord("2026-09-01", "INFY", 1520.0),
                PriceRecord("2026-09-02", "INFY", 1490.0),
            ],
            "TCS": [
                PriceRecord("2026-09-01", "TCS", 3800.0),
                PriceRecord("2026-09-02", "TCS", 3950.0),
            ],
        }
        rules = [
            LevelRule(id="r_infy", instrument="INFY", type="level", direction="below", level=1500.0),
            LevelRule(id="r_tcs", instrument="TCS", type="level", direction="above", level=3900.0),
        ]
        alerts = evaluate_rules(prices, rules)
        self.assertEqual(len(alerts), 2)
        # Both fired on 2026-09-02, sorted by rule id: 'r_infy' < 'r_tcs'
        self.assertEqual([a.rule_id for a in alerts], ["r_infy", "r_tcs"])

    def test_output_sorting_by_date_then_rule_id(self):
        """Alerts must be ordered strictly by date ascending, then rule ID ascending."""
        prices = {
            "A": [
                PriceRecord("2026-01-01", "A", 10.0),
                PriceRecord("2026-01-02", "A", 5.0),
                PriceRecord("2026-01-05", "A", 15.0),
            ],
            "B": [
                PriceRecord("2026-01-01", "B", 100.0),
                PriceRecord("2026-01-02", "B", 50.0),
                PriceRecord("2026-01-03", "B", 150.0),
            ],
        }
        rules = [
            LevelRule(id="z_rule", instrument="A", type="level", direction="below", level=8.0),   # fires 2026-01-02
            LevelRule(id="a_rule", instrument="B", type="level", direction="below", level=80.0),  # fires 2026-01-02
            LevelRule(id="m_rule", instrument="B", type="level", direction="above", level=120.0), # fires 2026-01-03
            LevelRule(id="b_rule", instrument="A", type="level", direction="above", level=12.0),  # fires 2026-01-05
        ]
        alerts = evaluate_rules(prices, rules)
        expected = [
            ("2026-01-02", "a_rule"),
            ("2026-01-02", "z_rule"),
            ("2026-01-03", "m_rule"),
            ("2026-01-05", "b_rule"),
        ]
        self.assertEqual([(a.date, a.rule_id) for a in alerts], expected)

    def test_no_alerts_fired(self):
        """When no conditions are met, evaluate_rules returns an empty list."""
        prices = {
            "INFY": [
                PriceRecord("2026-09-01", "INFY", 1520.0),
                PriceRecord("2026-09-02", "INFY", 1530.0),
            ]
        }
        rules = [LevelRule(id="r1", instrument="INFY", type="level", direction="below", level=1500.0)]
        alerts = evaluate_rules(prices, rules)
        self.assertEqual(len(alerts), 0)

    def test_rule_targeting_missing_instrument_noop(self):
        """
        Rules targeting instruments not present in the prices data should gracefully
        produce no alerts without error.
        """
        prices = {
            "INFY": [
                PriceRecord("2026-09-01", "INFY", 1520.0),
                PriceRecord("2026-09-02", "INFY", 1490.0),
            ]
        }
        rules = [
            LevelRule(id="r_aapl", instrument="AAPL", type="level", direction="below", level=150.0),
            MACrossRule(id="r_msft", instrument="MSFT", type="ma_cross", short=5, long=20, direction="above"),
        ]
        alerts = evaluate_rules(prices, rules)
        self.assertEqual(len(alerts), 0)

    def test_simultaneous_alerts_same_instrument_same_date(self):
        """
        When multiple rules fire for the same instrument on the same date,
        alerts must be deterministically sorted alphabetically by rule ID.
        """
        prices = {
            "INFY": [
                PriceRecord("2026-09-01", "INFY", 1500.0),
                PriceRecord("2026-09-02", "INFY", 1500.0),
                PriceRecord("2026-09-03", "INFY", 1600.0),
                PriceRecord("2026-09-04", "INFY", 1300.0),
            ]
        }
        # On 2026-09-04:
        # - rule_z_level fires (price was 1600 > 1500, now 1300 < 1500)
        # - rule_a_ma fires (MA2 was 1550 > MA3 1533.33, now MA2 1450 < MA3 1466.67)
        rules = [
            LevelRule(id="rule_z_level", instrument="INFY", type="level", direction="below", level=1500.0),
            MACrossRule(id="rule_a_ma", instrument="INFY", type="ma_cross", short=2, long=3, direction="below"),
        ]
        alerts = evaluate_rules(prices, rules)
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0].date, "2026-09-04")
        self.assertEqual(alerts[0].rule_id, "rule_a_ma")
        self.assertEqual(alerts[1].date, "2026-09-04")
        self.assertEqual(alerts[1].rule_id, "rule_z_level")



class TestValidationAndErrors(unittest.TestCase):
    """Tests file loading, malformed data, and rule parameter validations."""

    def _write_temp(self, content: str, suffix: str) -> str:
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=suffix) as tf:
            tf.write(content)
            return tf.name

    def test_missing_files(self):
        with self.assertRaises(PriceAlertError):
            load_prices("non_existent_prices.csv")
        with self.assertRaises(PriceAlertError):
            load_rules("non_existent_rules.json")

    def test_empty_prices_file(self):
        path = self._write_temp("", ".csv")
        try:
            with self.assertRaises(PriceAlertError):
                load_prices(path)
        finally:
            os.remove(path)

    def test_missing_csv_columns(self):
        path = self._write_temp("date,close\n2026-01-01,100\n", ".csv")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path)
            self.assertIn("missing required columns", str(ctx.exception))
        finally:
            os.remove(path)

    def test_malformed_csv_row(self):
        path = self._write_temp("date,instrument,close\n2026-01-01,INFY\n", ".csv")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path)
            self.assertIn("Malformed row", str(ctx.exception))
        finally:
            os.remove(path)

    def test_invalid_date_in_csv(self):
        path = self._write_temp("date,instrument,close\n2026-13-45,INFY,100\n", ".csv")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path)
            self.assertIn("Invalid date format", str(ctx.exception))
        finally:
            os.remove(path)

    def test_invalid_close_price_in_csv(self):
        path = self._write_temp("date,instrument,close\n2026-01-01,INFY,not_a_number\n", ".csv")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path)
            self.assertIn("Invalid close price", str(ctx.exception))
        finally:
            os.remove(path)

    def test_infinite_close_price_rejected(self):
        """Infinity and NaN values in close prices must be rejected with PriceAlertError."""
        for val in ["Infinity", "-Infinity", "inf", "-inf", "NaN", "nan"]:
            path = self._write_temp(f"date,instrument,close\n2026-01-01,INFY,{val}\n", ".csv")
            try:
                with self.assertRaises(PriceAlertError) as ctx:
                    load_prices(path)
                self.assertIn("must be a finite, non-negative number", str(ctx.exception))
            finally:
                os.remove(path)

    def test_missing_csv_fields_rejected(self):
        """Missing or empty values in any required CSV field must be rejected."""
        # Empty close
        path_close = self._write_temp("date,instrument,close\n2026-01-01,INFY,\n", ".csv")
        # Empty instrument
        path_inst = self._write_temp("date,instrument,close\n2026-01-01,,1500.0\n", ".csv")
        # Empty date
        path_date = self._write_temp("date,instrument,close\n,INFY,1500.0\n", ".csv")

        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path_close)
            self.assertIn("Invalid close price", str(ctx.exception))

            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path_inst)
            self.assertIn("Empty instrument name", str(ctx.exception))

            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path_date)
            self.assertIn("Invalid date format", str(ctx.exception))
        finally:
            os.remove(path_close)
            os.remove(path_inst)
            os.remove(path_date)


    def test_duplicate_date_same_instrument(self):
        path = self._write_temp(
            "date,instrument,close\n"
            "2026-01-01,INFY,100.0\n"
            "2026-01-01,INFY,105.0\n",
            ".csv"
        )
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_prices(path)
            self.assertIn("Duplicate price entry", str(ctx.exception))
        finally:
            os.remove(path)

    def test_malformed_json_rules(self):
        path = self._write_temp("{invalid json}", ".json")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_rules(path)
            self.assertIn("Malformed JSON", str(ctx.exception))
        finally:
            os.remove(path)

    def test_rules_not_a_list(self):
        path = self._write_temp('{"id": "r1"}', ".json")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_rules(path)
            self.assertIn("must contain a JSON array", str(ctx.exception))
        finally:
            os.remove(path)

    def test_duplicate_rule_id(self):
        data = [
            {"id": "r1", "instrument": "INFY", "type": "level", "direction": "below", "level": 1500},
            {"id": "r1", "instrument": "TCS", "type": "level", "direction": "above", "level": 3000},
        ]
        path = self._write_temp(json.dumps(data), ".json")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_rules(path)
            self.assertIn("Duplicate rule id", str(ctx.exception))
        finally:
            os.remove(path)

    def test_unknown_rule_type(self):
        data = [{"id": "r1", "instrument": "INFY", "type": "rsi_cross", "direction": "below"}]
        path = self._write_temp(json.dumps(data), ".json")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_rules(path)
            self.assertIn("unsupported type", str(ctx.exception))
        finally:
            os.remove(path)

    def test_invalid_direction(self):
        data = [{"id": "r1", "instrument": "INFY", "type": "level", "direction": "sideways", "level": 100}]
        path = self._write_temp(json.dumps(data), ".json")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_rules(path)
            self.assertIn("invalid direction", str(ctx.exception))
        finally:
            os.remove(path)

    def test_ma_cross_invalid_windows(self):
        # short >= long
        data = [{"id": "r1", "instrument": "INFY", "type": "ma_cross", "short": 20, "long": 5, "direction": "above"}]
        path = self._write_temp(json.dumps(data), ".json")
        try:
            with self.assertRaises(PriceAlertError) as ctx:
                load_rules(path)
            self.assertIn("short window < long window", str(ctx.exception))
        finally:
            os.remove(path)


class TestCLIExecution(unittest.TestCase):
    """End-to-end CLI execution tests."""

    def test_cli_sample_run(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = run_cli(["prices.csv", "rules.json"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        output_lines = stdout.getvalue().strip().splitlines()
        self.assertEqual(output_lines, [
            "2026-09-03 r1 INFY level below 1500",
            "2026-09-24 r2 TCS ma_cross 5/20 above",
        ])

    def test_cli_insufficient_arguments(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = run_cli(["prices.csv"])
        self.assertEqual(code, 1)
        self.assertIn("Usage:", stderr.getvalue())

    def test_cli_missing_file_reports_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = run_cli(["non_existent.csv", "rules.json"])
        self.assertEqual(code, 1)
        self.assertIn("Error:", stderr.getvalue())

    def test_cli_no_alerts_outputs_nothing(self):
        """When rules generate no alerts, CLI prints nothing and exits 0."""
        csv_path = "test_temp_no_alerts.csv"
        json_path = "test_temp_no_alerts.json"
        with open(csv_path, "w") as f:
            f.write("date,instrument,close\n2026-01-01,INFY,1600\n2026-01-02,INFY,1650\n")
        with open(json_path, "w") as f:
            f.write('[{"id": "r1", "instrument": "INFY", "type": "level", "direction": "below", "level": 1500}]\n')

        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = run_cli([csv_path, json_path])
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            if os.path.exists(json_path):
                os.remove(json_path)

    def test_csv_with_utf8_bom(self):
        """CSV files with UTF-8 BOM (e.g. from Excel) must parse seamlessly."""
        csv_path = "test_temp_bom.csv"
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write("date,instrument,close\n2026-09-01,INFY,1520.0\n2026-09-02,INFY,1490.0\n")
        try:
            prices = load_prices(csv_path)
            self.assertIn("INFY", prices)
            self.assertEqual(len(prices["INFY"]), 2)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_json_with_utf8_bom(self):
        """JSON rules files with UTF-8 BOM must parse seamlessly."""
        json_path = "test_temp_bom.json"
        with open(json_path, "w", encoding="utf-8-sig") as f:
            f.write('[{"id": "r1", "instrument": "INFY", "type": "level", "direction": "below", "level": 1500}]\n')
        try:
            rules = load_rules(json_path)
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].id, "r1")
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)

    def test_date_normalization_unpadded(self):
        """Unpadded date strings (e.g. 2026-9-2) must be normalized to canonical YYYY-MM-DD."""
        csv_path = "test_temp_norm_date.csv"
        with open(csv_path, "w") as f:
            f.write("date,instrument,close\n2026-9-2,INFY,1520.0\n2026-09-10,INFY,1490.0\n")
        try:
            prices = load_prices(csv_path)
            self.assertEqual(prices["INFY"][0].date, "2026-09-02")
            self.assertEqual(prices["INFY"][1].date, "2026-09-10")
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_case_insensitive_headers(self):
        """CSV headers with uppercase or mixed case must parse correctly."""
        csv_path = "test_temp_headers.csv"
        with open(csv_path, "w") as f:
            f.write("DATE,INSTRUMENT,CLOSE\n2026-09-01,INFY,1520.0\n")
        try:
            prices = load_prices(csv_path)
            self.assertIn("INFY", prices)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_brief_exact_5_row_snippet(self):
        """
        Tests the exact 5 rows printed in the brief's 'Inputs' snippet:
        INFY crosses below 1500 on 2026-09-03 (r1 fires).
        TCS has only 2 rows, so r2 (long window=20) cannot fire due to insufficient history.
        """
        csv_path = "test_temp_brief_snippet.csv"
        with open(csv_path, "w") as f:
            f.write(
                "date,instrument,close\n"
                "2026-09-01,INFY,1520.50\n"
                "2026-09-01,TCS,3890.00\n"
                "2026-09-02,INFY,1534.10\n"
                "2026-09-02,TCS,3902.25\n"
                "2026-09-03,INFY,1498.00\n"
            )
        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = run_cli([csv_path, "rules.json"])
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            output = stdout.getvalue().strip()
            self.assertEqual(output, "2026-09-03 r1 INFY level below 1500")
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
