#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import os
import sys
import time
import types
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# ---------------- JSON (fast) ----------------
def _get_json_dumps():
    try:
        import orjson  # type: ignore

        def dumps(obj: Any) -> str:
            return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")

        return dumps
    except Exception:
        pass

    def dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    return dumps


JSON_DUMPS = _get_json_dumps()

# ---------------- deps bootstrap ----------------
def _pip_install(pkgs: Sequence[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    subprocess.check_call(cmd)


def _ensure_deps(no_auto_pip: bool) -> None:
    needed_imports = [
        ("hhoppe_tools", ["hhoppe-tools"]),
        ("more_itertools", ["more-itertools"]),
        ("numba", ["numba"]),
        ("numpy", ["numpy"]),
    ]

    missing: List[str] = []
    for mod_name, pip_names in needed_imports:
        try:
            __import__(mod_name)
        except Exception:
            missing.extend(pip_names)

    if missing:
        if no_auto_pip:
            raise RuntimeError("Missing deps. Install:\n  pip install " + " ".join(missing))
        _pip_install(missing)


def _ensure_random32_py(no_auto_pip: bool, target_dir: Path) -> None:
    # blackjack.py makes `import random32` (random32.py in repo)
    try:
        __import__("random32")
        return
    except Exception:
        pass

    url = "https://raw.githubusercontent.com/hhoppe/blackjack/main/random32.py"
    out_path = target_dir / "random32.py"
    if out_path.exists():
        return

    if no_auto_pip:
        raise RuntimeError(
            "random32.py missing. Put random32.py next to blackjack.py "
            "or download it from:\n  " + url
        )

    import urllib.request

    with urllib.request.urlopen(url) as r:
        out_path.write_bytes(r.read())

# ---------------- AST strip (import-safe) ----------------
BANNED_TOPLEVEL_CALLS = {
    "monte_carlo_hand",
    "monte_carlo_hand_cpu",
    "monte_carlo_hand_gpu",
    "run_simulations",
}

BANNED_TOPLEVEL_NAMES = {"_value", "_value_sdv", "_value_std", "_played_hands"}


def _call_func_name(call: ast.Call) -> str | None:
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _contains_banned_call(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = _call_func_name(n)
            if name in BANNED_TOPLEVEL_CALLS:
                return True
    return False


def _contains_banned_name(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in BANNED_TOPLEVEL_NAMES:
            return True
    return False


class _StripTopLevelCalls(ast.NodeTransformer):
    """
    Делает blackjack.py импортируемым в воркерах:
    - удаляет top-level demo/самотесты (особенно Monte-Carlo и asserts)
    - НЕ трогает логику внутри функций/классов
    """

    def __init__(self) -> None:
        super().__init__()
        self._nest = 0

    def _enter(self) -> None:
        self._nest += 1

    def _exit(self) -> None:
        self._nest -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._enter()
        try:
            return self.generic_visit(node)
        finally:
            self._exit()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._enter()
        try:
            return self.generic_visit(node)
        finally:
            self._exit()

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self._enter()
        try:
            return self.generic_visit(node)
        finally:
            self._exit()

    def visit_Assert(self, node: ast.Assert) -> ast.AST | None:
        # all top-level asserts — self-test/demo, not required for export
        if self._nest == 0:
            return None
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if self._nest > 0:
            return node
        # module docstring
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node
        # drop top-level calls (print/demo)
        if isinstance(node.value, ast.Call):
            return None
        # drop expr referencing banned temp names
        if _contains_banned_name(node):
            return None
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if self._nest == 0 and (_contains_banned_call(node) or _contains_banned_name(node)):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if self._nest == 0 and (_contains_banned_call(node) or _contains_banned_name(node)):
            return None
        return self.generic_visit(node)

    def visit_If(self, node: ast.If) -> ast.AST | None:
        if self._nest == 0 and (_contains_banned_call(node) or _contains_banned_name(node)):
            return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.If)
        return node2 if (node2.body or node2.orelse) else None

    def visit_For(self, node: ast.For) -> ast.AST | None:
        if self._nest == 0 and (_contains_banned_call(node) or _contains_banned_name(node)):
            return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.For)
        return node2 if (node2.body or node2.orelse) else None

    def visit_While(self, node: ast.While) -> ast.AST | None:
        if self._nest == 0 and (_contains_banned_call(node) or _contains_banned_name(node)):
            return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.While)
        return node2 if (node2.body or node2.orelse) else None

    def visit_With(self, node: ast.With) -> ast.AST | None:
        if self._nest == 0 and (_contains_banned_call(node) or _contains_banned_name(node)):
            return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.With)
        return node2 if node2.body else None

    def visit_Try(self, node: ast.Try) -> ast.AST | None:
        if self._nest == 0 and (_contains_banned_call(node) or _contains_banned_name(node)):
            return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.Try)
        if not node2.handlers and not node2.finalbody:
            flat = node2.body + node2.orelse
            return flat if flat else None
        if not node2.body and not node2.orelse and not node2.handlers and not node2.finalbody:
            return None
        return node2


def load_hoppe_blackjack(blackjack_py: Path, effort: int) -> Any:
    blackjack_py = blackjack_py.resolve()
    if not blackjack_py.exists():
        raise FileNotFoundError(f"blackjack.py not found: {blackjack_py}")

    os.environ["EFFORT"] = str(int(effort))

    # local imports blackjack.py (random32.py etc.)
    sys.path.insert(0, str(blackjack_py.parent))
    try:
        src = blackjack_py.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(blackjack_py))
        tree2 = _StripTopLevelCalls().visit(tree)
        assert tree2 is not None
        ast.fix_missing_locations(tree2)

        mod_name = "hoppe_blackjack__isolated"
        mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = mod
        exec(compile(tree2, filename=str(blackjack_py), mode="exec"), mod.__dict__)
        return mod
    finally:
        if sys.path and sys.path[0] == str(blackjack_py.parent):
            sys.path.pop(0)

# ---------------- Export (sim-compatible structure) ----------------
UPCARDS: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

def _idx_from_upcard(u: int) -> int:
    return 9 if u == 11 else (u - 2)

def _idx_from_pair_value(v: int) -> int:
    return 9 if v == 11 else (v - 2)

def _payout_key(p: float) -> str:
    if p == 1.5:
        return "3to2"
    if p == 1.2:
        return "6to5"
    return "1to1"

def _dealer17_key(hit_soft17: bool) -> str:
    return "H17" if hit_soft17 else "S17"

def _das_key(das: bool) -> str:
    return "DAS" if das else "NDAS"

def _surrender_key(ls: bool) -> str:
    return "LS" if ls else "NS"

def _peek_key(obo: bool) -> str:
    return "PEEK" if obo else "ENHC"

def _double_rule_key(double_min_total: int) -> str:
    if double_min_total == 0:
        return "DBL-any"
    if double_min_total == 9:
        return "DBL-9-11"
    return "DBL-10-11"

def _split_key(split_to_num_hands: float) -> str:
    if split_to_num_hands == math.inf:
        return "SPLIT-INF"
    return f"SPLIT-{int(split_to_num_hands)}"

def _bool_key(name: str, v: bool) -> str:
    return name if v else f"N{name}"

def rules_to_key(rules: Any) -> str:
    decks = "INF" if rules.num_decks == math.inf else f"{int(rules.num_decks)}D"
    return "__".join(
        [
            decks,
            _dealer17_key(bool(rules.hit_soft17)),
            _das_key(bool(rules.double_after_split)),
            _surrender_key(bool(rules.late_surrender)),
            _peek_key(bool(rules.obo)),
            _payout_key(float(rules.blackjack_payout)),
            _double_rule_key(int(rules.double_min_total)),
            _split_key(float(rules.split_to_num_hands)),
            _bool_key("RSA", bool(rules.resplit_aces)),
            _bool_key("HSA", bool(rules.hit_split_aces)),
            _bool_key("DSA", bool(rules.double_split_aces)),
            f"CUT-{int(rules.cut_card)}",
            f"P{int(rules.num_players)}",
        ]
    )

def _wiki_house_edge_if_available(blackjack: Any, rules: Any, strategy: Any) -> Optional[float]:
    calc_cls = getattr(blackjack, "WikipediaHouseEdgeCalculator", None)
    if calc_cls is not None:
        try:
            return float(calc_cls()(rules, strategy))
        except Exception:
            return None
    edge_calcs = getattr(blackjack, "EDGE_CALCULATORS", None)
    if isinstance(edge_calcs, dict) and "wiki" in edge_calcs:
        try:
            return float(edge_calcs["wiki"](rules, strategy))
        except Exception:
            return None
    return None

def export_one(
    blackjack: Any,
    rules: Any,
    *,
    edge_mode: str,
    quiet: bool,
) -> Dict[str, Any]:
    strategy = blackjack.Strategy()
    tables = blackjack.basic_strategy_tables(rules, strategy)

    hard_arr = tables["hard"]
    soft_arr = tables["soft"]
    pair_arr = tables["pair"]

    idx_u = _idx_from_upcard

    hard: Dict[str, Dict[str, str]] = {}
    for total in range(5, 22):
        i = total - 5
        row: Dict[str, str] = {}
        for u in UPCARDS:
            row[str(u)] = str(hard_arr[i, idx_u(u)])
        hard[str(total)] = row

    soft: Dict[str, Dict[str, str]] = {}
    for total in range(13, 22):
        i = total - 13
        row2: Dict[str, str] = {}
        for u in UPCARDS:
            row2[str(u)] = str(soft_arr[i, idx_u(u)])
        soft[str(total)] = row2

    pair: Dict[str, Dict[str, str]] = {}
    for pv in range(2, 12):
        i = _idx_from_pair_value(pv)
        row3: Dict[str, str] = {}
        for u in UPCARDS:
            row3[str(u)] = str(pair_arr[i, idx_u(u)])
        pair[str(pv)] = row3

    house_edge: Optional[float] = None
    house_edge_pct: Optional[float] = None
    house_edge_source: Optional[str] = None
    house_edge_prob: Optional[float] = None

    if edge_mode != "skip":
        he_wiki = _wiki_house_edge_if_available(blackjack, rules, strategy) if edge_mode in ("auto", "wiki") else None
        if he_wiki is not None:
            house_edge = float(he_wiki)
            house_edge_pct = float(he_wiki * 100.0)
            house_edge_source = "wiki"
        else:
            if edge_mode == "wiki":
                house_edge = None
                house_edge_pct = None
                house_edge_source = "wiki"
            else:
                he = blackjack.probabilistic_house_edge(rules, strategy, quiet=quiet)
                house_edge = float(he)
                house_edge_pct = float(he * 100.0)
                house_edge_source = "probabilistic"

        # sim-compatible: houseEdgeProbabilistic field in meta.
        # To avoid double heavy calculation:
        if house_edge_source == "probabilistic" and house_edge is not None:
            house_edge_prob = float(house_edge)
        else:
            try:
                house_edge_prob = float(
                    blackjack.probabilistic_house_edge(rules, strategy, quiet=True)
                )
            except Exception:
                house_edge_prob = None

    payload: Dict[str, Any] = {
        "version": 2,
        "key": rules_to_key(rules),
        "rules": dataclasses.asdict(rules),
        "matrices": {"hard": hard, "soft": soft, "pair": pair},
        "houseEdge": house_edge,
        "houseEdgePct": house_edge_pct,
        "meta": {
            "source": "hhoppe/blackjack blackjack.py",
            "effortRequested": os.environ.get("EFFORT"),
            "effortUsed": int(getattr(blackjack, "EFFORT", -1)),
            "aceUpcard": 11,
            "houseEdgeSource": house_edge_source,
            "houseEdgeProbabilistic": house_edge_prob,
        },
    }
    return payload

# ---------------- Multiprocessing worker ----------------
_BJ: Any = None
_BASE_RULES: Any = None
_EDGE_MODE: str = "auto"
_QUIET: bool = True

TaskT = Tuple[float, bool, bool, bool, bool, float, int, float, bool, bool, bool, Optional[int], int]

def _worker_init(blackjack_py: str, effort: int, edge_mode: str, quiet: bool) -> None:
    global _BJ, _BASE_RULES, _EDGE_MODE, _QUIET
    _BJ = load_hoppe_blackjack(Path(blackjack_py), effort=effort)
    _BASE_RULES = _BJ.Rules()
    _EDGE_MODE = edge_mode
    _QUIET = quiet

def _make_rules_from_tuple(t: TaskT) -> Any:
    (
        num_decks,
        hit_soft17,
        das,
        late_surrender,
        obo,
        payout,
        double_min_total,
        split_to_num_hands,
        resplit_aces,
        hit_split_aces,
        double_split_aces,
        cut_card_override,
        num_players,
    ) = t

    assert _BJ is not None and _BASE_RULES is not None

    # IMPORTANT: like in sim if cutCard is not defined then pass -1 and crearte default.
    cut_card = -1 if cut_card_override is None else int(cut_card_override)

    overrides: Dict[str, Any] = {
        "num_decks": float(num_decks),
        "hit_soft17": bool(hit_soft17),
        "double_after_split": bool(das),
        "late_surrender": bool(late_surrender),
        "obo": bool(obo),
        "blackjack_payout": float(payout),
        "double_min_total": int(double_min_total),
        "split_to_num_hands": float(split_to_num_hands),
        "resplit_aces": bool(resplit_aces),
        "hit_split_aces": bool(hit_split_aces),
        "double_split_aces": bool(double_split_aces),
        "cut_card": int(cut_card),
        "num_players": int(num_players),
    }
    return dataclasses.replace(_BASE_RULES, **overrides)

def _worker_export(task: TaskT) -> str:
    assert _BJ is not None
    rules = _make_rules_from_tuple(task)
    payload = export_one(_BJ, rules, edge_mode=_EDGE_MODE, quiet=_QUIET)
    return JSON_DUMPS(payload)

# ---------------- Task generation ----------------
def _parse_decks_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        out.append(math.inf if p.upper() == "INF" else float(int(p)))
    return out

def _parse_split_list(s: str) -> List[float]:
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        if p in ("INF", "SPLIT-INF"):
            out.append(math.inf)
        elif p.startswith("SPLIT-"):
            out.append(float(int(p.split("-", 1)[1])))
        else:
            out.append(float(int(p)))
    return out

def _tasks_all(
    decks_list: Sequence[float],
    hit_soft17_list: Sequence[bool],
    das_list: Sequence[bool],
    ls_list: Sequence[bool],
    obo_list: Sequence[bool],
    payout_list: Sequence[float],
    dbl_list: Sequence[int],
    split_list: Sequence[float],
    rsa_list: Sequence[bool],
    hsa_list: Sequence[bool],
    dsa_list: Sequence[bool],
    cut_card_override: Optional[int],
    num_players: int,
) -> Iterator[TaskT]:
    for num_decks in decks_list:
        for hit_soft17 in hit_soft17_list:
            for das in das_list:
                for late_surrender in ls_list:
                    for obo in obo_list:
                        for payout in payout_list:
                            for double_min_total in dbl_list:
                                for split_to_num_hands in split_list:
                                    for resplit_aces in rsa_list:
                                        for hit_split_aces in hsa_list:
                                            for double_split_aces in dsa_list:
                                                yield (
                                                    float(num_decks),
                                                    bool(hit_soft17),
                                                    bool(das),
                                                    bool(late_surrender),
                                                    bool(obo),
                                                    float(payout),
                                                    int(double_min_total),
                                                    float(split_to_num_hands),
                                                    bool(resplit_aces),
                                                    bool(hit_split_aces),
                                                    bool(double_split_aces),
                                                    cut_card_override,
                                                    int(num_players),
                                                )

def _read_existing_keys(jsonl_path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("key")
                if isinstance(k, str):
                    keys.add(k)
            except Exception:
                continue
    return keys

# ---------------- Main ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blackjack-py", type=Path, required=True, help="Path to hhoppe/blackjack blackjack.py")

    ap.add_argument("--out", type=Path, required=True, help="Output JSONL file path (append with --resume)")
    ap.add_argument("--resume", action="store_true", help="Skip already-written keys in existing JSONL")

    ap.add_argument("--effort", type=int, default=1, help="EFFORT env for Hoppe code (0..4).")
    ap.add_argument("--edge", choices=["skip", "wiki", "prob", "auto"], default="auto",
                    help="House edge mode. auto=wiki if available else prob. skip=fastest")
    ap.add_argument("--quiet", action="store_true", help="Pass quiet=True into probabilistic house edge")

    ap.add_argument("--no-auto-pip", action="store_true", help="Do not pip install missing deps automatically")

    ap.add_argument("--workers", type=int, default=0, help="Num worker processes. 0=auto (cpu_count-1)")
    ap.add_argument("--maxtasksperchild", type=int, default=25,
                    help="Restart worker after N tasks (stability vs speed tradeoff)")

    ap.add_argument("--flush-every", type=int, default=200, help="Flush output every N lines")
    ap.add_argument("--progress-every", type=int, default=200, help="Progress report every N lines to stderr")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--one", action="store_true")

    ap.add_argument("--decks", type=str, default=None, help="Comma list: 1,2,4,6,8,INF")
    ap.add_argument("--dealer17", choices=["H17", "S17"], default=None)
    ap.add_argument("--das", choices=["DAS", "NDAS"], default=None)
    ap.add_argument("--surrender", choices=["LS", "NS"], default=None)
    ap.add_argument("--peek", choices=["PEEK", "ENHC"], default=None)
    ap.add_argument("--payout", choices=["3to2", "6to5", "1to1"], default=None)
    ap.add_argument("--doubleRule", choices=["DBL-any", "DBL-9-11", "DBL-10-11"], default=None)
    ap.add_argument("--split", type=str, default=None, help="Comma list: 0,2,4,INF or SPLIT-0,...")
    ap.add_argument("--rsa", choices=["RSA", "NRSA"], default=None)
    ap.add_argument("--hsa", choices=["HSA", "NHSA"], default=None)
    ap.add_argument("--dsa", choices=["DSA", "NDSA"], default=None)
    ap.add_argument("--cutCard", type=int, default=None, help="Override cut card integer. If omitted, pass -1 to Hoppe.")
    ap.add_argument("--players", type=int, default=1, help="Num players. Default 1.")

    args = ap.parse_args()

    _ensure_deps(no_auto_pip=args.no_auto_pip)
    _ensure_random32_py(no_auto_pip=args.no_auto_pip, target_dir=Path(args.blackjack_py).resolve().parent)

    decks_list = _parse_decks_list(args.decks) if args.decks else [1.0, 2.0, 4.0, 6.0, 8.0, math.inf]
    hit_soft17_list = [args.dealer17 == "H17"] if args.dealer17 else [True, False]
    das_list = [args.das == "DAS"] if args.das else [True, False]
    ls_list = [args.surrender == "LS"] if args.surrender else [True, False]
    obo_list = [args.peek == "PEEK"] if args.peek else [True, False]
    payout_list = (
        [1.5] if args.payout == "3to2" else
        [1.2] if args.payout == "6to5" else
        [1.0] if args.payout == "1to1" else
        [1.5, 1.2, 1.0]
    )
    dbl_list = (
        [0] if args.doubleRule == "DBL-any" else
        [9] if args.doubleRule == "DBL-9-11" else
        [10] if args.doubleRule == "DBL-10-11" else
        [0, 9, 10]
    )
    split_list = _parse_split_list(args.split) if args.split else [0.0, 2.0, 4.0, math.inf]
    rsa_list = [args.rsa == "RSA"] if args.rsa else [False, True]
    hsa_list = [args.hsa == "HSA"] if args.hsa else [False, True]
    dsa_list = [args.dsa == "DSA"] if args.dsa else [False, True]

    existing_keys: Set[str] = set()
    out_path = args.out.resolve()
    if args.resume:
        existing_keys = _read_existing_keys(out_path)
        print(f"[resume] loaded {len(existing_keys)} keys from {out_path}", file=sys.stderr)

    cpu = os.cpu_count() or 2
    workers = args.workers if args.workers > 0 else max(1, cpu - 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mode = "a" if (args.resume and out_path.exists()) else "w"
    f = out_path.open(out_mode, encoding="utf-8", buffering=1024 * 1024)

    if args.one:
        if len(decks_list) != 1:
            raise SystemExit("--one requires a single --decks value")
        tasks = list(
            _tasks_all(
                decks_list=decks_list,
                hit_soft17_list=hit_soft17_list[:1] if args.dealer17 else [True],
                das_list=das_list[:1] if args.das else [True],
                ls_list=ls_list[:1] if args.surrender else [True],
                obo_list=obo_list[:1] if args.peek else [True],
                payout_list=payout_list[:1] if args.payout else [1.5],
                dbl_list=dbl_list[:1] if args.doubleRule else [0],
                split_list=split_list[:1] if args.split else [4.0],
                rsa_list=rsa_list[:1] if args.rsa else [False],
                hsa_list=hsa_list[:1] if args.hsa else [False],
                dsa_list=dsa_list[:1] if args.dsa else [False],
                cut_card_override=args.cutCard,
                num_players=int(args.players),
            )
        )
        if len(tasks) != 1:
            raise SystemExit("--one produced != 1 task; set all flags explicitly")
        task_stream: Iterable[TaskT] = tasks
    else:
        task_stream = _tasks_all(
            decks_list=decks_list,
            hit_soft17_list=hit_soft17_list,
            das_list=das_list,
            ls_list=ls_list,
            obo_list=obo_list,
            payout_list=payout_list,
            dbl_list=dbl_list,
            split_list=split_list,
            rsa_list=rsa_list,
            hsa_list=hsa_list,
            dsa_list=dsa_list,
            cut_card_override=args.cutCard,
            num_players=int(args.players),
        )

    import multiprocessing as mp

    # macOS: stable spawn 
    ctx = mp.get_context("spawn")

    started = time.time()
    written = 0
    skipped = 0

    try:
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(str(args.blackjack_py), int(args.effort), str(args.edge), bool(args.quiet)),
            maxtasksperchild=int(args.maxtasksperchild),
        ) as pool:
            for line in pool.imap_unordered(_worker_export, task_stream, chunksize=8):
                if existing_keys:
                    try:
                        obj = json.loads(line)
                        k = obj.get("key")
                        if isinstance(k, str) and k in existing_keys:
                            skipped += 1
                            continue
                    except Exception:
                        pass

                f.write(line)
                f.write("\n")
                written += 1

                if args.flush_every > 0 and (written % int(args.flush_every) == 0):
                    f.flush()

                if args.progress_every > 0 and (written % int(args.progress_every) == 0):
                    dt = max(1e-9, time.time() - started)
                    rate = written / dt
                    print(
                        f"[progress] written={written} skipped={skipped} rate={rate:.2f} lines/s out={out_path.name}",
                        file=sys.stderr,
                    )
    finally:
        f.flush()
        f.close()

    dt = max(1e-9, time.time() - started)
    print(
        f"[done] written={written} skipped={skipped} seconds={dt:.1f} rate={written/dt:.2f} lines/s out={out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()