# Blackjack strategy table generator

This app generates Blackjack basic strategy tables based on [bjnb: Blackjack Notebook](https://github.com/hhoppe/blackjack). Simple viewer app: [app.21logic.com](https://app.21logic.com).

## How-to

### 1. Installation (macOS / Linux)
```
brew install python@3.12
```

```
cd /blackjack-basic-strategy-tables
deactivate 2>/dev/null || true
rm -rf .venv

/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install hhoppe-tools matplotlib more-itertools numba numpy tqdm
```

### 2. Clone blackjack.py
```
git clone https://github.com/hhoppe/blackjack.git
```

### Probabilistic analysis (effort 1)
export_tables_prob.py
```
python export_tables_prob.py \
--blackjack-py ./blackjack.py \
--out ./tables_prob.jsonl \
--all \
--edge auto \
--effort 1 \
--workers 8 \
--maxtasksperchild 25 \
--flush-every 500 \
--progress-every 500
```

### Probabilistic analysis (effort 3)
export_tables_sim.py
```
EFFORT=3 python export_hoppe_tables_v3_resume.py \
  --blackjack-py ./blackjack.py \
  --format jsonl \
  --out ./hoppe_tables_resume_3.jsonl \
  --all \
  --edge auto \
  --effort 3 --resume
```

## DB generation
```
python make_sql_dump.py --in ./tables_prob.jsonl --out ./tables_dump_prob.sql --truncate
```