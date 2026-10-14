"""Lab 5: propositional logic, CNF, resolution and DPLL.

Standard-library-only implementation. Notebook: Lab5_Logic.ipynb.
Models are dictionaries; clauses are frozensets of strings such as P and ~P.
"""

from itertools import combinations, product
import random


# %% Step 1: expression trees
def Sym(n):
    if not isinstance(n, str) or not n or n.startswith("~"):
        raise ValueError("A symbol needs a nonempty name without a leading ~")
    return ("sym", n)


def Not(a): return ("not", a)
def And(a, b): return ("and", a, b)
def Or(a, b): return ("or", a, b)
def Imp(a, b): return ("imp", a, b)
def Iff(a, b): return ("iff", a, b)


def symbols(s):
    if s[0] == "sym":
        return {s[1]}
    if s[0] == "not":
        return symbols(s[1])
    return symbols(s[1]) | symbols(s[2])


def to_str(s):
    op = s[0]
    if op == "sym":
        return s[1]
    if op == "not":
        return f"(~{to_str(s[1])})"
    signs = {"and": "&", "or": "|", "imp": "=>", "iff": "<=>"}
    return f"({to_str(s[1])} {signs[op]} {to_str(s[2])})"


def conjoin(sentences):
    """Combine a nonempty sequence of sentences into one binary And tree."""
    it = iter(sentences)
    try:
        result = next(it)
    except StopIteration as exc:
        raise ValueError("conjoin requires at least one sentence") from exc
    for sentence in it:
        result = And(result, sentence)
    return result


# %% Step 2: semantics
def pl_true(s, model):
    op = s[0]
    if op == "sym":
        return model[s[1]]
    if op == "not":
        return not pl_true(s[1], model)
    a, b = pl_true(s[1], model), pl_true(s[2], model)
    if op == "and": return a and b
    if op == "or": return a or b
    if op == "imp": return (not a) or b
    if op == "iff": return a == b
    raise ValueError(f"Unknown operator: {op}")


# %% Step 3: truth tables
def all_models(syms):
    """Sorted names; False precedes True, for reproducible counters."""
    names = sorted(set(syms))
    for values in product((False, True), repeat=len(names)):
        yield dict(zip(names, values))


def tt_entails(kb, alpha, max_models=None):
    """Return (entailed, models examined); None means the limit was exceeded."""
    if isinstance(kb, list):
        kb = conjoin(kb)
    count = 0
    for model in all_models(symbols(kb) | symbols(alpha)):
        count += 1
        if max_models is not None and count > max_models:
            return None, count - 1
        if pl_true(kb, model) and not pl_true(alpha, model):
            return False, count
    return True, count


def is_valid(s):
    return all(pl_true(s, m) for m in all_models(symbols(s)))


def is_satisfiable(s):
    return any(pl_true(s, m) for m in all_models(symbols(s)))


def print_truth_table(s):
    names = sorted(symbols(s))
    print(" | ".join(names + [to_str(s)]))
    for model in all_models(names):
        print(" | ".join(str(int(model[n])) for n in names)
              + " | " + str(int(pl_true(s, model))))


# %% Step 4: conversion to CNF
def eliminate_iff(s):
    op = s[0]
    if op == "sym": return s
    if op == "not": return Not(eliminate_iff(s[1]))
    a, b = eliminate_iff(s[1]), eliminate_iff(s[2])
    if op == "iff": return And(Imp(a, b), Imp(b, a))
    return (op, a, b)


def eliminate_imp(s):
    op = s[0]
    if op == "sym": return s
    if op == "not": return Not(eliminate_imp(s[1]))
    a, b = eliminate_imp(s[1]), eliminate_imp(s[2])
    if op == "imp": return Or(Not(a), b)
    if op == "iff":
        raise ValueError("Run eliminate_iff before eliminate_imp")
    return (op, a, b)


def push_not(s):
    op = s[0]
    if op == "sym": return s
    if op == "not":
        a = s[1]
        if a[0] == "sym": return s
        if a[0] == "not": return push_not(a[1])
        if a[0] == "and": return Or(push_not(Not(a[1])), push_not(Not(a[2])))
        if a[0] == "or": return And(push_not(Not(a[1])), push_not(Not(a[2])))
        raise ValueError("Remove implications before push_not")
    if op not in ("and", "or"):
        raise ValueError("Remove implications before push_not")
    return (op, push_not(s[1]), push_not(s[2]))


def distribute(s):
    if s[0] in ("sym", "not"): return s
    a, b = distribute(s[1]), distribute(s[2])
    if s[0] == "and": return And(a, b)
    if s[0] != "or": raise ValueError("distribute expects and/or and literals")
    if a[0] == "and":
        return And(distribute(Or(a[1], b)), distribute(Or(a[2], b)))
    if b[0] == "and":
        return And(distribute(Or(a, b[1])), distribute(Or(a, b[2])))
    return Or(a, b)


def to_cnf(s):
    return distribute(push_not(eliminate_imp(eliminate_iff(s))))


def opposite(literal):
    return literal[1:] if literal.startswith("~") else "~" + literal


def literal_symbol(literal):
    return literal[1:] if literal.startswith("~") else literal


def is_tautology_clause(clause):
    return any(opposite(lit) in clause for lit in clause)


def to_clauses(s):
    """Input is a CNF expression; remove duplicate and tautological clauses."""
    if s[0] == "and":
        return to_clauses(s[1]) | to_clauses(s[2])

    def literals(expr):
        if expr[0] == "sym": return {expr[1]}
        if expr[0] == "not" and expr[1][0] == "sym": return {"~" + expr[1][1]}
        if expr[0] == "or": return literals(expr[1]) | literals(expr[2])
        raise ValueError("to_clauses expects CNF; call to_cnf first")

    clause = frozenset(literals(s))
    return set() if is_tautology_clause(clause) else {clause}


def clausify(s):
    return to_clauses(to_cnf(s))


def clause_key(clause):
    return len(clause), tuple(sorted(clause))


def format_clauses(clauses):
    return " & ".join("{" + ", ".join(sorted(c)) + "}"
                      for c in sorted(clauses, key=clause_key)) or "TRUE (no clauses)"


# %% Step 5: resolution by refutation
def resolve(ci, cj):
    """Resolve one complementary pair at a time, discarding tautologies."""
    result = []
    for lit in sorted(ci):
        neg = opposite(lit)
        if neg in cj:
            resolvent = frozenset((set(ci) - {lit}) | (set(cj) - {neg}))
            if not is_tautology_clause(resolvent):
                result.append(resolvent)
    return result


def resolution_unsat(clauses, stats=None, max_resolvents=None):
    """True=UNSAT, False=SAT, None=limit; counts all retained resolvents.

    Full pair enumeration each round follows the manual. Already-known
    resolvents are counted too; tautological candidates are not counted.
    """
    stats = {} if stats is None else stats
    stats["resolvents"] = 0
    clauses = {frozenset(c) for c in clauses if not is_tautology_clause(c)}
    if frozenset() in clauses:
        return True
    while True:
        new = set()
        for ci, cj in combinations(sorted(clauses, key=clause_key), 2):
            for resolvent in resolve(ci, cj):
                stats["resolvents"] += 1
                if max_resolvents is not None and stats["resolvents"] > max_resolvents:
                    return None
                if not resolvent:
                    return True
                new.add(resolvent)
        if new <= clauses:
            return False
        clauses |= new


def pl_resolution(kb_clauses, alpha, stats=None, max_resolvents=None):
    return resolution_unsat(set(kb_clauses) | clausify(Not(alpha)),
                            stats, max_resolvents)


# %% Step 6: DPLL
def clause_value(clause, model):
    """True=satisfied, False=contradicted, None=still undecided."""
    unknown = False
    for lit in clause:
        name = literal_symbol(lit)
        if name not in model:
            unknown = True
        elif model[name] == (not lit.startswith("~")):
            return True
    return None if unknown else False


def find_pure_symbol(symbols, clauses, model):
    """Consider only clauses not already satisfied by the model."""
    active = [c for c in clauses if clause_value(c, model) is not True]
    present = set().union(*active) if active else set()
    for name in sorted(symbols):
        if name in model: continue
        positive, negative = name in present, "~" + name in present
        if positive != negative:
            return name, positive
    return None


def find_unit_clause(clauses, model):
    for clause in sorted(clauses, key=clause_key):
        if clause_value(clause, model) is True: continue
        remaining = [lit for lit in clause if literal_symbol(lit) not in model]
        if len(remaining) == 1:
            lit = remaining[0]
            return literal_symbol(lit), not lit.startswith("~")
    return None


def dpll(clauses, symbols, model, *, stats=None, use_unit=True, use_pure=True):
    stats = {"calls": 0} if stats is None else stats
    stats["calls"] = stats.get("calls", 0) + 1
    active = []
    for clause in clauses:
        value = clause_value(clause, model)
        if value is False: return False
        if value is None: active.append(clause)
    if not active: return dict(model)

    def assign(name, value):
        return dpll(clauses, set(symbols) - {name}, {**model, name: value},
                    stats=stats, use_unit=use_unit, use_pure=use_pure)

    forced = find_pure_symbol(symbols, active, model) if use_pure else None
    if forced is None and use_unit:
        forced = find_unit_clause(active, model)
    if forced is not None:
        return assign(*forced)
    name = sorted(symbols)[0]
    answer = assign(name, True)
    # {} is a successful model for the empty conjunction; never use `if answer`.
    if answer is not False: return answer
    return assign(name, False)


def dpll_satisfiable(clauses, stats=None, *, use_unit=True, use_pure=True):
    """Return a complete satisfying model or False; optional stats holds calls."""
    stats = {} if stats is None else stats
    stats["calls"] = 0
    clauses = tuple(sorted({frozenset(c) for c in clauses}, key=clause_key))
    names = {literal_symbol(lit) for c in clauses for lit in c}
    answer = dpll(clauses, names, {}, stats=stats,
                  use_unit=use_unit, use_pure=use_pure)
    if answer is False: return False
    # Unassigned names are irrelevant once every clause already has a true literal.
    return {name: answer.get(name, False) for name in sorted(names)}


# %% Step 7: Wumpus knowledge base
def wumpus_problem():
    P11, P12, P21, P22, P31, B11, B21 = map(
        Sym, ("P11", "P12", "P21", "P22", "P31", "B11", "B21"))
    kb = conjoin([
        Not(P11), Iff(B11, Or(P12, P21)),
        Iff(B21, Or(Or(P11, P22), P31)), Not(B11), B21,
    ])
    return kb, [Not(P12), Not(P21), Not(P22), Not(P31), Or(P22, P31)]


def compare_entailment(kb, alpha):
    tt, models = tt_entails(kb, alpha)
    rstats, dstats = {}, {}
    resolution = pl_resolution(clausify(kb), alpha, rstats)
    countermodel = dpll_satisfiable(clausify(And(kb, Not(alpha))), dstats)
    dpll_answer = countermodel is False
    assert tt == resolution == dpll_answer
    return {"query": to_str(alpha), "entailed": tt, "tt_models": models,
            "resolution_resolvents": rstats["resolvents"],
            "dpll_calls": dstats["calls"], "countermodel": countermodel}


# %% Step 8: reproducible SAT comparison
def random_3sat(n, m, seed=1):
    """Exactly m distinct clauses; local Random(seed) avoids global side effects.

    Names X01,...,X20 sort numerically. Sample three different variable indices;
    choose a positive/negative sign with choice((True, False)). Resample duplicates.
    """
    if n < 3 or not 0 <= m <= 8 * (n * (n - 1) * (n - 2) // 6):
        raise ValueError("Invalid n or m for distinct three-literal clauses")
    rng = random.Random(seed)
    clauses = set()
    while len(clauses) < m:
        indices = rng.sample(range(1, n + 1), 3)
        clauses.add(frozenset(("" if rng.choice((True, False)) else "~")
                              + f"X{i:02d}" for i in indices))
    return clauses


def tt_sat_clauses(clauses, max_models=None):
    """SAT by exhaustive models: return (True/False/None, count)."""
    clauses = tuple(sorted(clauses, key=clause_key))
    names = {literal_symbol(lit) for c in clauses for lit in c}
    count = 0
    for model in all_models(names):
        count += 1
        if max_models is not None and count > max_models:
            return None, count - 1
        if all(clause_value(c, model) is True for c in clauses):
            return True, count
    return False, count


def benchmark():
    rows = []
    kb, queries = wumpus_problem()
    for index in (0, 2, 4):
        result = compare_entailment(kb, queries[index])
        rows.append({"problem": "Wumpus: " + result["query"],
                     "task": "entailment", "tt_result": result["entailed"],
                     "tt_count": result["tt_models"],
                     "resolution_result": result["entailed"],
                     "resolution_count": result["resolution_resolvents"],
                     "dpll_result": result["entailed"],
                     "dpll_count": result["dpll_calls"]})
    for n in (10, 15, 20):
        clauses = random_3sat(n, 4*n, seed=1)
        tt, models = tt_sat_clauses(clauses, max_models=200_000)
        rstats, dstats = {}, {}
        unsat = resolution_unsat(clauses, rstats, max_resolvents=100_000)
        model = dpll_satisfiable(clauses, dstats)
        sat = model is not False
        resolution_sat = None if unsat is None else not unsat
        assert tt is None or tt == sat
        assert resolution_sat is None or resolution_sat == sat
        if sat: assert all(clause_value(c, model) is True for c in clauses)
        rows.append({"problem": f"3-SAT n={n}, m={4*n}, seed=1", "task": "SAT",
                     "tt_result": tt, "tt_count": models,
                     "resolution_result": resolution_sat,
                     "resolution_count": rstats["resolvents"],
                     "dpll_result": sat, "dpll_count": dstats["calls"]})
    return rows


# %% Bonus exercises
def knights_problem():
    KA, KB, KC = map(Sym, ("KA", "KB", "KC"))
    return conjoin([Iff(KA, Not(KB)), Iff(KB, Iff(KA, KC)), Iff(KC, KB)])


def heuristic_comparison():
    """Four DPLL variants on all six comparison problems."""
    kb, queries = wumpus_problem()
    problems = [("Wumpus: " + to_str(queries[i]),
                 clausify(And(kb, Not(queries[i])))) for i in (0, 2, 4)]
    problems += [(f"3-SAT n={n}", random_3sat(n, 4*n, 1)) for n in (10, 15, 20)]
    rows = []
    for name, clauses in problems:
        row = {"problem": name}
        outcomes = []
        for label, unit, pure in (("both", True, True), ("unit_only", True, False),
                                  ("pure_only", False, True), ("neither", False, False)):
            stats = {}
            model = dpll_satisfiable(clauses, stats, use_unit=unit, use_pure=pure)
            outcomes.append(model is not False)
            if model is not False:
                assert all(clause_value(c, model) is True for c in clauses)
            row[label] = stats["calls"]
        assert len(set(outcomes)) == 1
        rows.append(row)
    return rows


if __name__ == "__main__":
    import json
    kb, queries = wumpus_problem()
    print(json.dumps({"wumpus": [compare_entailment(kb, q) for q in queries],
                      "benchmark": benchmark(), "heuristics": heuristic_comparison(),
                      "knights_models": [m for m in all_models(symbols(knights_problem()))
                                         if pl_true(knights_problem(), m)]}, indent=2))
