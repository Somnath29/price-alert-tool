# Price Alert Tool

A Python CLI that detects price-level crossings and moving-average (MA) crossovers from CSV prices using JSON rules.

## 1. How to Run

**Requirements:** Python 3.8+, standard library only.

```bash
python3 alerts.py prices.csv rules.json
python3 -m unittest -v test_alerts.py
```

**Expected sample output:**

```text
2026-09-03 r1 INFY level below 1500
2026-09-24 r2 TCS ma_cross 5/20 above
```

**Verification:** 43 tests passed. No output is printed when no alerts trigger.

### Manual Testing

Run these commands from the project directory.

**1. Test manual price and rule data**

```bash
python3 alerts.py testingData/manual_prices.csv testingData/manual_rules.json
```

**2. Test invalid price data**

```bash
python3 alerts.py testingData/invalid_prices.csv testingData/manual_rules.json
```

**3. Test invalid rule configuration**

```bash
python3 alerts.py testingData/manual_prices.csv testingData/invalid_rules.json
```

The invalid-input tests verify that the program handles incorrect price data and rule configurations appropriately.

## 2. AI Assistant Usage

I used AI to scaffold the CSV/JSON-based CLI, refine state-transition logic, and generate tests for unordered dates, boundary cases, moving-average history, invalid inputs, and output sorting.

## 3. What the Assistant Got Wrong, What I Changed, and Why

- **Repeated alerts:** Initial logic could alert every day a price remained below a threshold. I added previous/current-side tracking so alerts fire on crossings, not sustained conditions.
- **Boundary and MA handling:** Initial logic could treat a threshold touch as a crossing and detect an MA crossover before a prior relationship existed. I added three-state boundary handling (`+1`, `-1`, `0`) and use the first observation with both averages available to establish the baseline.

These changes prevent false alerts and ensure a crossover is detected only when supported by the observed data.