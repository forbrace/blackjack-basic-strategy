#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

UPCARDS: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def _install_random32_stub() -> None:
    if "random32" in sys.modules:
        return

    mod = types.ModuleType("random32")

    class Random32(random.Random):
        pass

    mod.Random32 = Random32  # type: ignore[attr-defined]
    mod.Random = random.Random  # type: ignore[attr-defined]
    mod.seed = random.seed  # type: ignore[attr-defined]
    mod.random = random.random  # type: ignore[attr-defined]
    mod.randint = random.randint  # type: ignore[attr-defined]
    mod.randrange = random.randrange  # type: ignore[attr-defined]
    mod.choice = random.choice  # type: ignore[attr-defined]
    sys.modules["random32"] = mod


class _StripTopLevelCalls(ast.NodeTransformer):
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

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.AST]:
        if self._nest > 0:
            return node
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node
        if isinstance(node.value, ast.Call):
            return None
        return node

    def visit_With(self, node: ast.With) -> Optional[ast.AST]:
        if self._nest > 0:
            return node
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.With)
        return node2 if node2.body else None

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Optional[ast.AST]:
        if self._nest > 0:
            return node
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.AsyncWith)
        return node2 if node2.body else None

    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.If)
        return node2 if (node2.body or node2.orelse) else None

    def visit_For(self, node: ast.For) -> Optional[ast.AST]:
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.For)
        return node2 if (node2.body or node2.orelse) else None

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Optional[ast.AST]:
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.AsyncFor)
        return node2 if (node2.body or node2.orelse) else None

    def visit_While(self, node: ast.While) -> Optional[ast.AST]:
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.While)
        return node2 if (node2.body or node2.orelse) else None

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Optional[ast.AST]:
        if self._nest > 0:
            return node
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.ExceptHandler)
        return node2 if node2.body else None

    def visit_Try(self, node: ast.Try) -> Optional[ast.AST]:
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.Try)

        if not node2.handlers and not node2.finalbody:
            flat = node2.body + node2.orelse
            return flat if flat else None

        if not node2.body and not node2.orelse and not node2.handlers and not node2.finalbody:
            return None

        return node2

def _install_numba_cuda_stub() -> None:
    """
    macOS usually has no CUDA. Hoppe's blackjack.py imports:
      from numba import cuda
      import numba.cuda.random
    This stub makes those imports succeed while reporting cuda.is_available() == False.

    It does NOT change CPU numba usage (numba.njit / numba.jit etc).
    """
    # If numba itself isn't installed, we don't try to fake full numba here.
    # Install numba (CPU) via pip/conda; we only stub CUDA parts.
    try:
        import numba as _numba  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "numba is not installed (CPU). Install it first: pip install numba"
        ) from e

    import types

    class _CudaUnsupported(RuntimeError):
        pass

    def _not_supported(*_a: object, **_k: object) -> None:
        raise _CudaUnsupported("CUDA is not available in this environment (macOS).")

    # ---- numba.cuda ----
    if "numba.cuda" not in sys.modules:
        cuda_mod = types.ModuleType("numba.cuda")
        cuda_mod.is_available = lambda: False  # type: ignore[attr-defined]
        cuda_mod.detect = _not_supported  # type: ignore[attr-defined]

        # If blackjack.py defines GPU kernels with @cuda.jit, make it a no-op decorator
        # so import doesn't fail. Those GPU kernels won't be used by our exporter.
        def _jit_noop(*args: object, **kwargs: object):  # noqa: ARG001
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]
            def _wrap(fn):
                return fn
            return _wrap

        cuda_mod.jit = _jit_noop  # type: ignore[attr-defined]
        cuda_mod.to_device = _not_supported  # type: ignore[attr-defined]
        cuda_mod.device_array = _not_supported  # type: ignore[attr-defined]
        cuda_mod.device_array_like = _not_supported  # type: ignore[attr-defined]

        sys.modules["numba.cuda"] = cuda_mod
        setattr(_numba, "cuda", cuda_mod)

    # ---- numba.cuda.random ----
    if "numba.cuda.random" not in sys.modules:
        rand_mod = types.ModuleType("numba.cuda.random")
        rand_mod.create_xoroshiro128p_states = _not_supported  # type: ignore[attr-defined]
        rand_mod.xoroshiro128p_uniform_float32 = _not_supported  # type: ignore[attr-defined]
        rand_mod.xoroshiro128p_uniform_float64 = _not_supported  # type: ignore[attr-defined]
        sys.modules["numba.cuda.random"] = rand_mod

        # attach as attribute for convenience
        cuda_mod2 = sys.modules["numba.cuda"]
        setattr(cuda_mod2, "random", rand_mod)

def load_hoppe_blackjack(blackjack_py: Path, *, effort_override: Optional[int]) -> Any:
    blackjack_py = blackjack_py.resolve()
    if not blackjack_py.exists():
        raise FileNotFoundError(f"blackjack.py not found: {blackjack_py}")

    _install_random32_stub()
    _install_numba_cuda_stub()

    prev_effort = os.environ.get("EFFORT")
    if effort_override is not None:
        os.environ["EFFORT"] = str(int(effort_override))

    try:
        src = blackjack_py.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(blackjack_py))
        tree2 = _StripTopLevelCalls().visit(tree)
        assert tree2 is not None
        ast.fix_missing_locations(tree2)

        mod_name = "hoppe_blackjack"
        mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = mod
        exec(compile(tree2, filename=str(blackjack_py), mode="exec"), mod.__dict__)
        return mod
    finally:
        if effort_override is not None:
            if prev_effort is None:
                os.environ.pop("EFFORT", None)
            else:
                os.environ["EFFORT"] = prev_effort


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


def _idx_from_upcard(u: int) -> int:
    return 9 if u == 11 else (u - 2)


def _idx_from_pair_value(v: int) -> int:
    return 9 if v == 11 else (v - 2)


def _wiki_house_edge_if_available(blackjack: Any, rules: Any, strategy: Any) -> Optional[float]:
    calc_cls = getattr(blackjack, "WikipediaHouseEdgeCalculator", None)
    if calc_cls is not None:
        return calc_cls()(rules, strategy)
    edge_calcs = getattr(blackjack, "EDGE_CALCULATORS", None)
    if isinstance(edge_calcs, dict) and "wiki" in edge_calcs:
        return edge_calcs["wiki"](rules, strategy)
    return None


def export_one(blackjack: Any, rules: Any, *, compute_edge: bool, quiet: bool) -> Dict[str, Any]:
    strategy = blackjack.Strategy()
    tables = blackjack.basic_strategy_tables(rules, strategy)

    hard_arr = tables["hard"]
    soft_arr = tables["soft"]
    pair_arr = tables["pair"]

    hard: Dict[str, Dict[str, str]] = {}
    for total in range(5, 22):
        i = total - 5
        row: Dict[str, str] = {}
        for u in UPCARDS:
            row[str(u)] = str(hard_arr[i, _idx_from_upcard(u)])
        hard[str(total)] = row

    soft: Dict[str, Dict[str, str]] = {}
    for total in range(13, 22):
        i = total - 13
        row2: Dict[str, str] = {}
        for u in UPCARDS:
            row2[str(u)] = str(soft_arr[i, _idx_from_upcard(u)])
        soft[str(total)] = row2

    pair: Dict[str, Dict[str, str]] = {}
    for pv in range(2, 12):
        i = _idx_from_pair_value(pv)
        row3: Dict[str, str] = {}
        for u in UPCARDS:
            row3[str(u)] = str(pair_arr[i, _idx_from_upcard(u)])
        pair[str(pv)] = row3

    house_edge: Optional[float] = None
    house_edge_pct: Optional[float] = None
    house_edge_source: Optional[str] = None
    house_edge_prob: Optional[float] = None

    if compute_edge:
        he_wiki = _wiki_house_edge_if_available(blackjack, rules, strategy)
        if he_wiki is not None:
            house_edge = float(he_wiki)
            house_edge_pct = float(he_wiki * 100.0)
            house_edge_source = "wiki"
        else:
            he = blackjack.probabilistic_house_edge(rules, strategy, quiet=quiet)
            house_edge = float(he)
            house_edge_pct = float(he * 100.0)
            house_edge_source = "probabilistic"

        try:
            house_edge_prob = float(blackjack.probabilistic_house_edge(rules, strategy, quiet=True))
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


def _payload_to_arrays(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    mats = payload["matrices"]
    hard = np.empty((17, 10), dtype=object)
    soft = np.empty((9, 10), dtype=object)
    pair = np.empty((10, 10), dtype=object)

    hard_m = mats["hard"]
    for total in range(5, 22):
        i = total - 5
        row = hard_m[str(total)]
        for u in UPCARDS:
            hard[i, _idx_from_upcard(u)] = row[str(u)]

    soft_m = mats["soft"]
    for total in range(13, 22):
        i = total - 13
        row = soft_m[str(total)]
        for u in UPCARDS:
            soft[i, _idx_from_upcard(u)] = row[str(u)]

    pair_m = mats["pair"]
    for pv in range(2, 12):
        i = _idx_from_pair_value(pv)
        row = pair_m[str(pv)]
        for u in UPCARDS:
            pair[i, _idx_from_upcard(u)] = row[str(u)]

    return {"hard": hard, "soft": soft, "pair": pair}


def _verify_export_is_correct(blackjack: Any, rules: Any, payload: Dict[str, Any], *, quiet: bool) -> None:
    strategy = blackjack.Strategy()
    tables = blackjack.basic_strategy_tables(rules, strategy)

    restored = _payload_to_arrays(payload)
    for name in ("hard", "soft", "pair"):
        got = restored[name]
        exp = tables[name]
        if got.shape != exp.shape:
            raise RuntimeError(f"VERIFY failed: shape mismatch for {name}: {got.shape} vs {exp.shape}")
        if not np.array_equal(got, exp):
            diff = np.argwhere(got != exp)
            i, j = diff[0].tolist()
            raise RuntimeError(
                f"VERIFY failed: table '{name}' mismatch at [{i},{j}]: got={got[i,j]!r} expected={exp[i,j]!r}"
            )

    if payload.get("houseEdge") is not None:
        he_json = float(payload["houseEdge"])
        src = str(payload.get("meta", {}).get("houseEdgeSource") or "")
        if src == "wiki":
            he_wiki = _wiki_house_edge_if_available(blackjack, rules, strategy)
            if he_wiki is None:
                raise RuntimeError("VERIFY failed: meta says wiki but wikipedia calculator returned None")
            if abs(he_json - float(he_wiki)) > 1e-15:
                raise RuntimeError(f"VERIFY failed: houseEdge(wiki) mismatch: json={he_json} expected={float(he_wiki)}")
        else:
            he2 = blackjack.probabilistic_house_edge(rules, strategy, quiet=quiet)
            if abs(he_json - float(he2)) > 1e-12:
                raise RuntimeError(f"VERIFY failed: houseEdge(prob) mismatch: json={he_json} expected={float(he2)}")


def _default_rules_like(blackjack: Any, *, num_decks: float, hit_soft17: bool) -> Any:
    r0 = blackjack.Rules()
    return dataclasses.replace(r0, num_decks=num_decks, hit_soft17=hit_soft17, cut_card=-1)


def _maybe_verify_expected_tables(blackjack: Any, rules: Any) -> None:
    expected_name: Optional[str] = None
    if rules.num_decks == 6 and rules.hit_soft17 is True:
        expected_name = "EXPECTED_BASIC_STRATEGY_ACTION_6DECKS_H17"
    elif rules.num_decks == 6 and rules.hit_soft17 is False:
        expected_name = "EXPECTED_BASIC_STRATEGY_ACTION_6DECKS_S17"
    elif rules.num_decks == 1 and rules.hit_soft17 is True:
        expected_name = "EXPECTED_BASIC_STRATEGY_ACTION_1DECK_H17"
    elif rules.num_decks == 1 and rules.hit_soft17 is False:
        expected_name = "EXPECTED_BASIC_STRATEGY_ACTION_1DECK_S17"

    if expected_name is None:
        return

    expected = getattr(blackjack, expected_name, None)
    if expected is None:
        return

    r_def = _default_rules_like(blackjack, num_decks=float(rules.num_decks), hit_soft17=bool(rules.hit_soft17))
    if dataclasses.asdict(rules) != dataclasses.asdict(r_def):
        return

    verify_fn = getattr(blackjack, "verify_action_tables", None)
    if callable(verify_fn):
        verify_fn(rules, expected=expected)
        return

    tables = blackjack.basic_strategy_tables(rules, blackjack.Strategy())
    for name, table in tables.items():
        code = np.array2string(table, formatter=dict(all=lambda x: f"{x:3}"))
        if code != expected[name]:
            raise RuntimeError(f"VERIFY-EXPECTED failed: Table '{name}' mismatch for {expected_name}.")


def _parse_decks_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        out.append(math.inf if p.upper() == "INF" else float(int(p)))
    return out


def _parse_split_value(s: str) -> float:
    s2 = s.strip().upper()
    if s2 == "SPLIT-INF":
        return math.inf
    if not s2.startswith("SPLIT-"):
        raise ValueError(f"Invalid --split: {s}")
    return float(int(s2.split("-", 1)[1]))


def _write_payload_files(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{payload['key']}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_payload_jsonl(payload: Dict[str, Any]) -> None:
    # canonical-ish: no indent, stable keys, compact separators
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blackjack-py", type=Path, required=True, help="Path to hhoppe blackjack.py")

    ap.add_argument("--out", type=Path, required=True, help="Output directory for --format files, or '-' for jsonl stdout.")
    ap.add_argument("--format", choices=["files", "jsonl"], default="files")

    ap.add_argument("--edge", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-expected", action="store_true")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--one", action="store_true")
    mode.add_argument("--all", action="store_true")

    ap.add_argument("--decks", type=str)
    ap.add_argument("--dealer17", choices=["H17", "S17"])
    ap.add_argument("--das", choices=["DAS", "NDAS"])
    ap.add_argument("--surrender", choices=["LS", "NS"])
    ap.add_argument("--peek", choices=["PEEK", "ENHC"])
    ap.add_argument("--payout", choices=["3to2", "6to5", "1to1"])
    ap.add_argument("--doubleRule", choices=["DBL-any", "DBL-9-11", "DBL-10-11"])
    ap.add_argument("--split", type=str)
    ap.add_argument("--rsa", choices=["RSA", "NRSA"])
    ap.add_argument("--hsa", choices=["HSA", "NHSA"])
    ap.add_argument("--dsa", choices=["DSA", "NDSA"])
    ap.add_argument("--cutCard", type=int, default=None)
    ap.add_argument("--players", type=int, default=None)

    args = ap.parse_args()

    jsonl_mode = args.format == "jsonl"
    out_is_stdout = str(args.out) == "-"

    if jsonl_mode and not out_is_stdout:
        # still allow a path, but we do not write there in jsonl mode
        pass
    if (not jsonl_mode) and out_is_stdout:
        raise SystemExit("--out '-' is only valid with --format jsonl")

    effort_override = 2 if args.verify_expected else None
    blackjack = load_hoppe_blackjack(args.blackjack_py, effort_override=effort_override)
    base_rules = blackjack.Rules()

    def emit(payload: Dict[str, Any]) -> None:
        if jsonl_mode:
            _write_payload_jsonl(payload)
        else:
            _write_payload_files(args.out, payload)

    if args.one:
        decks_vals = _parse_decks_list(args.decks) if args.decks else [float(base_rules.num_decks)]
        if len(decks_vals) != 1:
            raise SystemExit("--one requires exactly one deck value.")
        overrides: Dict[str, Any] = {"num_decks": decks_vals[0]}

        overrides["cut_card"] = -1 if args.cutCard is None else int(args.cutCard)

        if args.dealer17 is not None:
            overrides["hit_soft17"] = (args.dealer17 == "H17")
        if args.das is not None:
            overrides["double_after_split"] = (args.das == "DAS")
        if args.surrender is not None:
            overrides["late_surrender"] = (args.surrender == "LS")
        if args.peek is not None:
            overrides["obo"] = (args.peek == "PEEK")
        if args.payout is not None:
            overrides["blackjack_payout"] = 1.5 if args.payout == "3to2" else 1.2 if args.payout == "6to5" else 1.0
        if args.doubleRule is not None:
            overrides["double_min_total"] = 0 if args.doubleRule == "DBL-any" else 9 if args.doubleRule == "DBL-9-11" else 10
        if args.split is not None:
            overrides["split_to_num_hands"] = _parse_split_value(args.split)
        if args.rsa is not None:
            overrides["resplit_aces"] = (args.rsa == "RSA")
        if args.hsa is not None:
            overrides["hit_split_aces"] = (args.hsa == "HSA")
        if args.dsa is not None:
            overrides["double_split_aces"] = (args.dsa == "DSA")
        if args.players is not None:
            overrides["num_players"] = int(args.players)

        rules = dataclasses.replace(base_rules, **overrides)
        payload = export_one(blackjack, rules, compute_edge=args.edge, quiet=args.quiet)

        if args.verify:
            _verify_export_is_correct(blackjack, rules, payload, quiet=args.quiet)
        if args.verify_expected:
            _maybe_verify_expected_tables(blackjack, rules)

        emit(payload)
        return

    decks_list = _parse_decks_list(args.decks) if args.decks else [1.0, 2.0, 4.0, 6.0, 8.0, math.inf]
    hit_soft17_list = [args.dealer17 == "H17"] if args.dealer17 else [True, False]
    das_list = [args.das == "DAS"] if args.das else [True, False]
    ls_list = [args.surrender == "LS"] if args.surrender else [True, False]
    obo_list = [args.peek == "PEEK"] if args.peek else [True, False]
    payout_list = (
        [1.5] if args.payout == "3to2" else [1.2] if args.payout == "6to5" else [1.0]
        if args.payout else [1.5, 1.2, 1.0]
    )
    dbl_list = (
        [0] if args.doubleRule == "DBL-any" else [9] if args.doubleRule == "DBL-9-11" else [10]
        if args.doubleRule else [0, 9, 10]
    )
    split_list = [_parse_split_value(args.split)] if args.split else [0.0, 2.0, 4.0, math.inf]
    rsa_list = [args.rsa == "RSA"] if args.rsa else [False, True]
    hsa_list = [args.hsa == "HSA"] if args.hsa else [False, True]
    dsa_list = [args.dsa == "DSA"] if args.dsa else [False, True]

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
                                                overrides2: Dict[str, Any] = {
                                                    "num_decks": num_decks,
                                                    "hit_soft17": hit_soft17,
                                                    "double_after_split": das,
                                                    "late_surrender": late_surrender,
                                                    "obo": obo,
                                                    "blackjack_payout": payout,
                                                    "double_min_total": double_min_total,
                                                    "split_to_num_hands": split_to_num_hands,
                                                    "resplit_aces": resplit_aces,
                                                    "hit_split_aces": hit_split_aces,
                                                    "double_split_aces": double_split_aces,
                                                }
                                                overrides2["cut_card"] = -1 if args.cutCard is None else int(args.cutCard)
                                                if args.players is not None:
                                                    overrides2["num_players"] = int(args.players)

                                                rules2 = dataclasses.replace(base_rules, **overrides2)
                                                payload2 = export_one(blackjack, rules2, compute_edge=args.edge, quiet=args.quiet)

                                                if args.verify:
                                                    _verify_export_is_correct(blackjack, rules2, payload2, quiet=args.quiet)
                                                if args.verify_expected:
                                                    _maybe_verify_expected_tables(blackjack, rules2)

                                                emit(payload2)


if __name__ == "__main__":
    main()