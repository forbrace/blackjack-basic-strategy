#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

UPCARDS: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# --- Hoppe top-level demo killers ---
BANNED_TOPLEVEL_CALLS = {
    "monte_carlo_hand",
    "monte_carlo_hand_cpu",
    "monte_carlo_hand_gpu",
    "run_simulations",
}
BANNED_TOPLEVEL_NAMES = {"_value", "_value_sdv", "_value_std", "_played_hands"}


def _call_func_name(call: ast.Call) -> Optional[str]:
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
    """
    Makes blackjack.py importable:
       - removes top-level demo/self-tests that run Monte-Carlo and Pool on import
       - removes tails that depend on demo block temporary variables
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

    def visit_Assert(self, node: ast.Assert) -> Optional[ast.AST]:
        # all top-level assert (demo/self-test) are not needed for the exporter
        if self._nest == 0:
            return None
        return node

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.AST]:
        if self._nest > 0:
            return node
        # module docstring
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node
        # drop any top-level call expression
        if isinstance(node.value, ast.Call):
            return None
        # drop any expression referencing banned temp names
        if _contains_banned_name(node):
            return None
        return node

    def visit_Assign(self, node: ast.Assign) -> Optional[ast.AST]:
        if self._nest == 0:
            if _contains_banned_call(node) or _contains_banned_name(node):
                return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Optional[ast.AST]:
        if self._nest == 0:
            if _contains_banned_call(node) or _contains_banned_name(node):
                return None
        return self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        if self._nest == 0:
            if _contains_banned_call(node) or _contains_banned_name(node):
                return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.If)
        return node2 if (node2.body or node2.orelse) else None

    def visit_For(self, node: ast.For) -> Optional[ast.AST]:
        if self._nest == 0:
            if _contains_banned_call(node) or _contains_banned_name(node):
                return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.For)
        return node2 if (node2.body or node2.orelse) else None

    def visit_While(self, node: ast.While) -> Optional[ast.AST]:
        if self._nest == 0:
            if _contains_banned_call(node) or _contains_banned_name(node):
                return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.While)
        return node2 if (node2.body or node2.orelse) else None

    def visit_With(self, node: ast.With) -> Optional[ast.AST]:
        if self._nest == 0:
            if _contains_banned_call(node) or _contains_banned_name(node):
                return None
        node2 = self.generic_visit(node)
        assert isinstance(node2, ast.With)
        return node2 if node2.body else None

    def visit_Try(self, node: ast.Try) -> Optional[ast.AST]:
        if self._nest == 0:
            if _contains_banned_call(node) or _contains_banned_name(node):
                return None
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
    macOS usually does not have CUDA. Hoppe imports:
          from numba import cuda
          import numba.cuda.random
    We create a placeholder so that the import works, but cuda.is_available() == False.
    """
    try:
        import numba as _numba  # type: ignore
    except Exception as e:
        raise RuntimeError("numba is not installed. Install it: pip install numba") from e

    class _CudaUnsupported(RuntimeError):
        pass

    def _not_supported(*_a: object, **_k: object) -> None:
        raise _CudaUnsupported("CUDA is not available in this environment.")

    if "numba.cuda" not in sys.modules:
        cuda_mod = types.ModuleType("numba.cuda")
        cuda_mod.is_available = lambda: False  # type: ignore[attr-defined]
        cuda_mod.detect = _not_supported  # type: ignore[attr-defined]

        def _jit_noop(*args: object, **kwargs: object):  # noqa: ARG001
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def _wrap(fn: Any) -> Any:
                return fn

            return _wrap

        cuda_mod.jit = _jit_noop  # type: ignore[attr-defined]
        cuda_mod.to_device = _not_supported  # type: ignore[attr-defined]
        cuda_mod.device_array = _not_supported  # type: ignore[attr-defined]
        cuda_mod.device_array_like = _not_supported  # type: ignore[attr-defined]

        sys.modules["numba.cuda"] = cuda_mod
        setattr(_numba, "cuda", cuda_mod)

    if "numba.cuda.random" not in sys.modules:
        rand_mod = types.ModuleType("numba.cuda.random")
        rand_mod.create_xoroshiro128p_states = _not_supported  # type: ignore[attr-defined]
        rand_mod.xoroshiro128p_uniform_float32 = _not_supported  # type: ignore[attr-defined]
        rand_mod.xoroshiro128p_uniform_float64 = _not_supported  # type: ignore[attr-defined]
        sys.modules["numba.cuda.random"] = rand_mod

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
    # In the key, this is your historical label.
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


def _prob_house_edge(blackjack: Any, rules: Any, strategy: Any, quiet: bool) -> float:
    he = blackjack.probabilistic_house_edge(rules, strategy, quiet=quiet)
    return float(he)


def _call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(*args, **kwargs)
    supported: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(*args, **supported)


def _mc_house_edge_if_available(
    blackjack: Any,
    rules: Any,
    strategy: Any,
    *,
    mc_hands: int,
    mc_tasks: int,
) -> Optional[float]:
    # 1) EDGE_CALCULATORS["mc"] if available
    edge_calcs = getattr(blackjack, "EDGE_CALCULATORS", None)
    if isinstance(edge_calcs, dict) and "mc" in edge_calcs:
        try:
            fn = edge_calcs["mc"]
            val = _call_with_supported_kwargs(fn, rules, strategy, num_hands=mc_hands, num_tasks=mc_tasks)
            return float(val)
        except Exception:
            pass

    # 2) monte_carlo_house_edge if available
    fn2 = getattr(blackjack, "monte_carlo_house_edge", None)
    if callable(fn2):
        try:
            val2 = _call_with_supported_kwargs(fn2, rules, strategy, num_hands=mc_hands, num_tasks=mc_tasks)
            return float(val2)
        except Exception:
            pass

    return None


def export_one(
    blackjack: Any,
    rules: Any,
    *,
    edge_mode: str,  # skip|auto|wiki|prob|mc
    quiet: bool,
    mc_hands: int,
    mc_tasks: int,
) -> Dict[str, Any]:
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

    if edge_mode != "skip":
        if edge_mode in ("auto", "wiki"):
            he_wiki = _wiki_house_edge_if_available(blackjack, rules, strategy)
            if he_wiki is not None:
                house_edge = float(he_wiki)
                house_edge_pct = float(he_wiki * 100.0)
                house_edge_source = "wiki"
            elif edge_mode == "wiki":
                house_edge = None
                house_edge_pct = None
                house_edge_source = "wiki-missing"

        if house_edge is None and edge_mode in ("auto", "prob"):
            he = _prob_house_edge(blackjack, rules, strategy, quiet=quiet)
            house_edge = float(he)
            house_edge_pct = float(he * 100.0)
            house_edge_source = "probabilistic"

        if house_edge is None and edge_mode == "mc":
            he_mc = _mc_house_edge_if_available(
                blackjack, rules, strategy, mc_hands=int(mc_hands), mc_tasks=int(mc_tasks)
            )
            if he_mc is None:
                raise RuntimeError("Monte-Carlo edge is not available in this blackjack.py build.")
            house_edge = float(he_mc)
            house_edge_pct = float(he_mc * 100.0)
            house_edge_source = "montecarlo"

        try:
            house_edge_prob = float(_prob_house_edge(blackjack, rules, strategy, quiet=True))
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
            "mcHands": int(mc_hands) if edge_mode == "mc" else None,
            "mcTasks": int(mc_tasks) if edge_mode == "mc" else None,
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

        # Normalize both sides to string arrays to avoid dtype/enum/object mismatch.
        got_s = got.astype(str)
        exp_s = np.asarray(exp).astype(str)

        if got_s.shape != exp_s.shape:
            raise RuntimeError(f"VERIFY failed: shape mismatch for {name}: {got_s.shape} vs {exp_s.shape}")

        if not np.array_equal(got_s, exp_s):
            diff = np.argwhere(got_s != exp_s)
            i, j = diff[0].tolist()
            raise RuntimeError(
                f"VERIFY failed: table '{name}' mismatch at [{i},{j}]: got={got_s[i,j]!r} expected={exp_s[i,j]!r}"
            )

    # Keep houseEdge verification only if present and requested in payload.
    if payload.get("houseEdge") is not None:
        he_json = float(payload["houseEdge"])
        src = str(payload.get("meta", {}).get("houseEdgeSource") or "")
        if src == "wiki":
            he_wiki = _wiki_house_edge_if_available(blackjack, rules, strategy)
            if he_wiki is None:
                raise RuntimeError("VERIFY failed: meta says wiki but wikipedia calculator returned None")
            if abs(he_json - float(he_wiki)) > 1e-15:
                raise RuntimeError(f"VERIFY failed: houseEdge(wiki) mismatch: json={he_json} expected={float(he_wiki)}")
        elif src == "probabilistic":
            he2 = _prob_house_edge(blackjack, rules, strategy, quiet=quiet)
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


def _iter_tasks(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    # Deterministic order: same nesting, same lists.
    # This order MUST stay stable for strict resume.

    base_decks = [1.0, 2.0, 4.0, 6.0, 8.0, math.inf]
    decks_list = _parse_decks_list(args.decks) if args.decks else base_decks

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

    cut_card = -1 if args.cutCard is None else int(args.cutCard)
    num_players = int(args.players) if args.players is not None else None

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
                                                d: Dict[str, Any] = {
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
                                                }
                                                if num_players is not None:
                                                    d["num_players"] = int(num_players)
                                                yield d


def _strict_resume_prefix(
    blackjack: Any,
    base_rules: Any,
    args: argparse.Namespace,
    *,
    jsonl_path: Path,
    resume_from_line: Optional[int],
) -> int:
    """
    Проверяет, что существующий jsonl — это префикс ожидаемой последовательности.
    Возвращает индекс (кол-во строк), с которого продолжать.
    """
    if not jsonl_path.exists():
        return 0

    want_lines: Optional[int] = int(resume_from_line) if resume_from_line is not None else None
    got_lines = 0

    task_iter = _iter_tasks(args)

    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                got_key = obj.get("key")
                if not isinstance(got_key, str):
                    raise RuntimeError("bad jsonl line: missing key")
            except Exception as e:
                raise RuntimeError(f"bad jsonl line at {got_lines + 1}: {e}") from e

            try:
                overrides = next(task_iter)
            except StopIteration as e:
                raise RuntimeError("jsonl has more lines than possible combinations") from e

            rules = dataclasses.replace(base_rules, **overrides)
            exp_key = rules_to_key(rules)

            if got_key != exp_key:
                raise RuntimeError(
                    "resume prefix mismatch at line "
                    f"{got_lines + 1}:\n"
                    f"  file key: {got_key}\n"
                    f"  exp  key: {exp_key}\n"
                    "Stop to avoid skipping combinations."
                )

            got_lines += 1
            if want_lines is not None and got_lines >= want_lines:
                break

    if want_lines is not None:
        if got_lines != want_lines:
            raise RuntimeError(f"--resume-from-line {want_lines} requested, but file has only {got_lines} usable lines.")
        return want_lines

    return got_lines

def _verify_existing_jsonl(
    blackjack: Any,
    base_rules: Any,
    args: argparse.Namespace,
    *,
    jsonl_path: Path,
    quiet: bool,
) -> int:
    if not jsonl_path.exists():
        return 0

    task_iter = _iter_tasks(args)
    checked = 0
    record_no = 0  # counts non-empty JSON records

    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record_no += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"bad jsonl at file line {line_no} (record {record_no}): {e}") from e

            got_key = obj.get("key")
            if not isinstance(got_key, str):
                raise RuntimeError(f"bad jsonl at file line {line_no} (record {record_no}): missing key")

            try:
                overrides = next(task_iter)
            except StopIteration as e:
                raise RuntimeError("jsonl has more records than possible combinations") from e

            rules = dataclasses.replace(base_rules, **overrides)
            exp_key = rules_to_key(rules)
            if got_key != exp_key:
                raise RuntimeError(
                    f"VERIFY-EXISTING failed at file line {line_no} (record {record_no}):\n"
                    f"  file key: {got_key}\n"
                    f"  exp  key: {exp_key}"
                )

            # Verify tables only; do not require houseEdge reproducibility.
            obj2 = dict(obj)
            obj2["houseEdge"] = None
            _verify_export_is_correct(blackjack, rules, obj2, quiet=quiet)

            checked += 1

    return checked

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blackjack-py", type=Path, required=True, help="Path to hhoppe blackjack.py")

    ap.add_argument(
        "--effort",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Hoppe EFFORT (overrides env; higher = more accurate, slower).",
    )

    ap.add_argument("--out", type=Path, required=True, help="Output dir (files) or JSONL file path (jsonl).")
    ap.add_argument("--format", choices=["files", "jsonl"], default="files")

    ap.add_argument("--edge", choices=["skip", "auto", "wiki", "prob", "mc"], default="auto")
    ap.add_argument("--mc-hands", type=int, default=10_000_000, help="Monte-Carlo hands (only for --edge mc).")
    ap.add_argument("--mc-tasks", type=int, default=0, help="Monte-Carlo tasks/workers inside Hoppe (if supported).")

    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-expected", action="store_true")
    ap.add_argument("--verify-existing", action="store_true",
                help="Verify existing JSONL lines (keys + tables) before continuing.")

    ap.add_argument("--resume", action="store_true", help="Append to existing JSONL and strictly verify prefix.")
    ap.add_argument("--resume-from-line", type=int, default=None, help="Strict resume from exact line count (prefix).")

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

    if args.format == "files" and args.resume:
        raise SystemExit("--resume is only for --format jsonl")

    if args.format == "files" and args.resume_from_line is not None:
        raise SystemExit("--resume-from-line is only for --format jsonl")

    if args.format == "jsonl":
        out_path = args.out
    else:
        out_path = args.out

#     effort_override = 2 if args.verify_expected else None
#     blackjack = load_hoppe_blackjack(args.blackjack_py, effort_override=effort_override)
    effort_override = int(args.effort) if args.effort is not None else (2 if args.verify_expected else None)
    blackjack = load_hoppe_blackjack(args.blackjack_py, effort_override=effort_override)
    base_rules = blackjack.Rules()

    if args.one:
        # --one: enforce a single combination by restricting lists to exactly one value.
        decks_vals = _parse_decks_list(args.decks) if args.decks else [float(base_rules.num_decks)]
        if len(decks_vals) != 1:
            raise SystemExit("--one requires exactly one --decks value.")
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
        payload = export_one(
            blackjack,
            rules,
            edge_mode=str(args.edge),
            quiet=bool(args.quiet),
            mc_hands=int(args.mc_hands),
            mc_tasks=int(args.mc_tasks),
        )

        if args.verify:
            _verify_export_is_correct(blackjack, rules, payload, quiet=bool(args.quiet))
        if args.verify_expected:
            _maybe_verify_expected_tables(blackjack, rules)

        if args.format == "jsonl":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if args.resume else "w"
            with out_path.open(mode, encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            _write_payload_files(Path(args.out), payload)

        return

    # --all
    if args.format == "jsonl":
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.verify_existing:
            n = _verify_existing_jsonl(
                blackjack, base_rules, args, jsonl_path=out_path, quiet=bool(args.quiet)
            )
            print(f"[verify-existing] ok: {n} lines", file=sys.stderr)

        start_index = 0
        if args.resume:
            start_index = _strict_resume_prefix(
                blackjack,
                base_rules,
                args,
                jsonl_path=out_path,
                resume_from_line=args.resume_from_line,
            )
            print(f"[resume] prefix ok, continuing from line {start_index + 1}", file=sys.stderr)

        file_mode = "a" if args.resume else "w"
        with out_path.open(file_mode, encoding="utf-8") as f:
            idx = 0
            for overrides in _iter_tasks(args):
                if idx < start_index:
                    idx += 1
                    continue

                rules = dataclasses.replace(base_rules, **overrides)
                payload = export_one(
                    blackjack,
                    rules,
                    edge_mode=str(args.edge),
                    quiet=bool(args.quiet),
                    mc_hands=int(args.mc_hands),
                    mc_tasks=int(args.mc_tasks),
                )

                if args.verify:
                    _verify_export_is_correct(blackjack, rules, payload, quiet=bool(args.quiet))
                if args.verify_expected:
                    _maybe_verify_expected_tables(blackjack, rules)

                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                idx += 1

        return

    # files mode
    out_dir = Path(args.out)
    for overrides in _iter_tasks(args):
        rules = dataclasses.replace(base_rules, **overrides)
        payload = export_one(
            blackjack,
            rules,
            edge_mode=str(args.edge),
            quiet=bool(args.quiet),
            mc_hands=int(args.mc_hands),
            mc_tasks=int(args.mc_tasks),
        )

        if args.verify:
            _verify_export_is_correct(blackjack, rules, payload, quiet=bool(args.quiet))
        if args.verify_expected:
            _maybe_verify_expected_tables(blackjack, rules)

        _write_payload_files(out_dir, payload)


if __name__ == "__main__":
    main()