# Blackjack basic strategy tables generator

This repository generates **blackjack basic strategy tables** for many rule combinations and exports them as data files that can be shipped to a UI or loaded into a database.

The tables are produced using the [bjnb: Blackjack Notebook](https://github.com/hhoppe/blackjack) engine (from `hhoppe/blackjack`), which computes optimal basic-strategy action tables under different blackjack rules. 

A simple viewer app for these tables is available at [app.21logic.com](https://app.21logic.com).


## What you get in the end

You run the exporter scripts and obtain:

- **JSONL with generated tables** (one record per ruleset/table).
- Optionally: a **SQL dump** built from that JSONL, so you can import the same data into a database.

Example outputs used in this repo README:
- `tables_effort_3.jsonl` (higher-effort probabilistic export)
- `tables_dump_prob.sql` (SQL dump generated from JSONL)


## Why this exists

Basic strategy depends on rules (decks, H17/S17, DAS, surrender, peek/no-peek, split limits, etc.). A “one-size chart” is not enough if you want:
- a rules-accurate chart generator,
- consistent charts across many rule packs,
- a database-backed library of tables,
- a web UI that can instantly display the correct chart.

This repo is the **data pipeline** part: generate + export + (optionally) dump to SQL.


## Repository contents

- `export_tables_prob_effort_1.py`  
  Exports tables using **probabilistic analysis** at “effort 1” (faster / lower compute).

- `export_tables_prob_effort_3.py`  
  Exports tables using **probabilistic analysis** at “effort 3” (slower / more compute). The sample command uses `--verify` and `--resume`. 

- `make_sql_dump.py`  
  Converts a JSONL export into a **SQL dump**, with an optional `--truncate` mode mentioned in the repo README.


## Quick start

### 1) Install (macOS / Linux)

```
brew install python@3.12
```

The repo README uses Python 3.12 venv setup and installs:
```
cd /blackjack-basic-strategy-tables
deactivate 2>/dev/null || true
rm -rf .venv

/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install hhoppe-tools matplotlib more-itertools numba numpy tqdm
pip install ipython
```

### 2) Clone the upstream engine

Clone `hhoppe/blackjack` (the bjnb engine) next to this repo.
```
git clone https://github.com/hhoppe/blackjack.git
```

### 3) Export tables (probabilistic)

#### Effort 1 (fast baseline)

Run `export_tables_prob_effort_1.py` as shown below.
```
python export_tables_prob_effort_1.py \
--blackjack-py ./blackjack.py \
--out ./tables_effort_1.jsonl \
--all \
--edge auto \
--effort 1 \
--workers 8 \
--maxtasksperchild 25 \
--flush-every 500 \
--progress-every 500
```

#### Effort 3 (more compute)
```
EFFORT=3 python export_tables_prob_effort_3.py \
  --blackjack-py ./blackjack.py \
  --format jsonl \
  --out ./tables_effort_3.jsonl \
  --all \
  --edge prob \
  --verify \
  --effort 3 --resume
```

The practical difference:
- **effort 1** is meant for speed and iteration,
- **effort 3** is meant for higher-effort generation (more compute), useful when you want a stronger “production” dataset.


## Database dump generation

After you have a JSONL export, convert it to SQL:
```
python make_sql_dump.py --in ./tables_prob.jsonl --out ./tables_dump_prob.sql --truncate
```

This produces a SQL file you can import into your DB (schema/DDL details are defined by `make_sql_dump.py`).


## app.21logic.com (viewer) — short overview

This repo links to app.21logic.com as a “simple viewer app” for the generated tables.

In practice, this is where the exported tables become user-facing:
- pick a ruleset (or a “rule pack”),
- show the matching basic strategy charts (hard totals / soft totals / pairs),
- keep the UI fast by serving precomputed tables rather than recomputing on every request.

### Example charts

![Hard/Soft/Pairs example](https://app.21logic.com/images/blackjack_-6d,-h17,-das,-ls,-obo,-3_2,-dbl-any,-split-4,-nrsa,-nhsa,-ndsa,-cut-234,-p1-strategy.png)


## Notes

- The underlying engine (`bjnb`) supports many rule variations and produces optimal action tables for basic strategy. 
- This repo focuses on **exporting** those tables into formats that are easy to ship (JSONL) and store (SQL dump).