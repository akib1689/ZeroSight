"""Generate the employees evaluation set (99 questions).

Trap-dense composition calibrated for baseline difficulty (target
~80-85% on glm-5.3-flash; easy floor 18):

- boundary (16): strict-inequality date/salary bounds, period-inclusive
  counts, band complements — every trap generated only where the
  boundary value is populated, so the naive inclusive (or year-overflow)
  filter provably differs
- exclusion (11): total-minus-filtered counting incl. senior-title,
  decade, and multi-constraint complements
- disambiguation (9): stated-qualifier, entity-referenced ("the one who
  works in the same department as Y"), and double disambiguation
- comparison / arithmetic / verification / deep / rank / fanout /
  nested / history / crosscompare: near-miss booleans with strict-date
  boundaries, percentage/ratio chains, tenure checks, multi-hop chains,
  ranked exclusions and top-n set logic, argmax over group_by/headcount
  artifacts, derived thresholds chained into subsequent filters,
  penultimate-record lookups

Design constraints (answerable in BOTH modes by construction):
- entity selection flows through search arguments; placeholder chains
  stay within one artifact per hop (no multi-artifact joins, no date
  arithmetic, no branching a name on an unseen boolean)
- name-selection answers come from a single tool output (top-n list,
  sort over one artifact); booleans only where compare-expressible
Output: benchmarks/employees/tasks.json
"""

import datetime as dt
import json
import os
import sys

import httpx

BASE = os.environ.get("EMPLOYEES_API_URL", "http://127.0.0.1:8000")
client = httpx.Client(base_url=BASE, timeout=30.0)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "employees", "tasks.json")

FOREVER = "9999-01-01"
TITLES = [
    "Engineer",
    "Senior Engineer",
    "Staff",
    "Senior Staff",
    "Technique Leader",
    "Assistant Engineer",
    "Manager",
]


def api(method: str, path: str, **kw):
    resp = client.request(method, path, **kw)
    resp.raise_for_status()
    if resp.status_code == 204:
        return None
    return resp.json()


def search(**params) -> dict:
    params = {k: v for k, v in params.items() if v is not None}
    return api("GET", "/api/employees", params=params)


def stats(**params):
    params = {k: v for k, v in params.items() if v is not None}
    return api("GET", "/api/stats/salaries", params=params)


def top(**params):
    params = {k: v for k, v in params.items() if v is not None}
    return api("GET", "/api/stats/top-earners", params=params)


def dept_map() -> dict[str, str]:
    rows = api("GET", "/api/departments")
    return {r["dept_name"]: r["dept_no"] for r in rows}


def dept_names() -> list[str]:
    return list(dept_map())


def emp_detail(emp_no: int) -> dict:
    return api("GET", f"/api/employees/{emp_no}")


def salary_history(emp_no: int) -> list[dict]:
    return api("GET", f"/api/employees/{emp_no}/salaries",
               params={"from_date": "1900-01-01", "to_date": "9999-12-31"})


def title_history(emp_no: int) -> list[dict]:
    return api("GET", f"/api/employees/{emp_no}/titles",
               params={"from_date": "1900-01-01", "to_date": "9999-12-31"})


def dept_history(emp_no: int) -> list[dict]:
    return api("GET", f"/api/employees/{emp_no}/departments",
               params={"from_date": "1900-01-01", "to_date": "9999-12-31"})


def salary_at(emp_no: int, at: str):
    rows = api("GET", f"/api/employees/{emp_no}/salaries", params={"at_date": at})
    return rows[0]["salary"] if rows else None


def title_at(emp_no: int, at: str):
    rows = api("GET", f"/api/employees/{emp_no}/titles", params={"at_date": at})
    return rows[0]["title"] if rows else None


def covers(row: dict, at: str) -> bool:
    upper = row["to_date"] if row["to_date"] != FOREVER else "9999-12-31"
    return row["from_date"] <= at < upper


def full(item: dict) -> str:
    return f"{item['first_name']} {item['last_name']}"


def unique_named(item: dict) -> bool:
    hit = search(first_name=item["first_name"], last_name=item["last_name"], page_size=1)
    return hit["total"] == 1


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def fingerprint_check():
    total = search(page_size=1)["total"]
    if total != 300024:
        print(f"WARNING: employee total is {total}, expected 300024 — DB not pristine?",
              file=sys.stderr)


def find_anchors():
    pool = []
    page = 1
    while len(pool) < 80:
        data = search(page=page, page_size=100, sort="emp_no")
        if not data["items"]:
            break
        for row in data["items"]:
            if not row.get("active"):
                continue
            hit = search(first_name=row["first_name"], last_name=row["last_name"], page_size=1)
            if hit["total"] == 1 and hit["items"][0].get("active"):
                pool.append(hit["items"][0])
                if len(pool) >= 80:
                    break
        page += 1

    raised, promoted, transferred = [], [], []
    for row in pool:
        if len(raised) >= 3 and len(promoted) >= 3 and len(transferred) >= 3:
            break
        sal = salary_history(row["emp_no"])
        tit = title_history(row["emp_no"])
        dep = dept_history(row["emp_no"])
        if len(raised) < 3 and len(sal) >= 2:
            raised.append({"row": row, "sal": sal})
        if len(promoted) < 3 and len(tit) >= 2 and tit[0]["title"] != tit[1]["title"]:
            promoted.append({"row": row, "tit": tit})
        if len(transferred) < 3 and len(dep) >= 2 and len({d["dept_name"] for d in dep}) >= 2:
            transferred.append({"row": row, "dep": dep})
    return pool, raised, promoted, transferred


def find_ambiguous(pool: list[dict], need: int) -> list[dict]:
    """Names shared by 2+ employees where a qualifier (dept/title/gender)
    narrows the API search to exactly one LIKE-match."""
    out: list[dict] = []
    seen_first: set[str] = set()
    for row in pool:
        if len(out) >= need:
            break
        fn = row["first_name"]
        if fn in seen_first:
            continue
        seen_first.add(fn)
        page = search(first_name=fn, page_size=100)
        groups: dict[tuple[str, str], list[dict]] = {}
        for it in page["items"]:
            groups.setdefault((it["first_name"], it["last_name"]), []).append(it)
        for (fn2, ln2), items in groups.items():
            if len(items) < 2 or len(out) >= need:
                continue
            for target in items:
                if not target.get("active") or any(
                    o["row"]["emp_no"] == target["emp_no"] for o in out
                ):
                    continue
                quals = []
                if target.get("dept_no"):
                    quals.append({
                        "kind": "dept",
                        "params": {"dept_no": target["dept_no"]},
                        "phrase": f"who works in the {target['dept_name']} department",
                    })
                if target.get("title"):
                    quals.append({
                        "kind": "title",
                        "params": {"title": target["title"]},
                        "phrase": f"whose current title is {target['title']}",
                    })
                if target.get("gender"):
                    g = "female" if target["gender"] == "F" else "male"
                    quals.append({"kind": "gender", "params": {"gender": target["gender"]},
                                  "phrase": f"who is {g}"})
                working = []
                for q in quals:
                    hit = search(first_name=fn2, last_name=ln2, page_size=100, **q["params"])
                    if hit["total"] == 1 and hit["items"][0]["emp_no"] == target["emp_no"]:
                        working.append(q)
                if working:
                    out.append({"row": target, "quals": working})
    return out




def _day_shift(iso: str, days: int) -> str:
    """ISO date shifted by N days."""
    return (dt.date.fromisoformat(iso) + dt.timedelta(days=days)).isoformat()


def generate():
    fingerprint_check()
    pool, raised, promoted, transferred = find_anchors()
    dm = dept_map()
    names = dept_names()
    tasks: list[dict] = []

    def add(category, question, gold, meta=None):
        tasks.append({
            "id": f"emp-{len(tasks) + 1:03d}",
            "question": question,
            "gold_answer": fmt(gold),
            "category": category,
            "metadata": meta or {},
        })

    def _cnt(**p):
        return search(page_size=1, **p)["total"]

    mgr_info = {dn: api("GET", f"/api/departments/{dm[dn]}")["manager"] for dn in names}
    mgr_sal = {dn: emp_detail(m["emp_no"])["salary"] for dn, m in mgr_info.items()}

    # ── A: easy floor (18) ───────────────────────────────────────────────
    a = pool
    add("lookup", f"What is the current annual salary of {full(a[0])}?", a[0]["salary"],
        {"emp_no": a[0]["emp_no"]})
    add("lookup", f"What is the current job title of {full(a[1])}?", a[1]["title"],
        {"emp_no": a[1]["emp_no"]})
    add("lookup", f"In which department does {full(a[2])} currently work? Give the department "
        f"name.", a[2]["dept_name"], {"emp_no": a[2]["emp_no"]})
    add("lookup", f"On what date was {full(a[3])} hired? Answer in YYYY-MM-DD format.",
        a[3]["hire_date"], {"emp_no": a[3]["emp_no"]})

    d0 = emp_detail(a[0]["emp_no"])
    mgr0 = api("GET", f"/api/departments/{d0['department']['dept_no']}")
    add("multihop", f"Who is the current manager of the department in which {full(a[0])} "
        f"works? Give the manager's full name.", full(mgr0["manager"]),
        {"dept_no": mgr0["dept_no"], "manager_emp_no": mgr0["manager"]["emp_no"]})
    add("multihop", "What is the current annual salary of the manager of the Development "
        "department?", mgr_sal["Development"],
        {"manager_emp_no": mgr_info["Development"]["emp_no"]})
    top_f = top(dept_no=dm["Finance"], n=1)[0]
    add("multihop", "Who is the top earner in the Finance department? Give the full name.",
        full(top_f), {"emp_no": top_f["emp_no"]})
    add("multihop", "How many Engineers currently work in the Development department?",
        _cnt(dept_no=dm["Development"], title="Engineer"), {"dept_no": dm["Development"]})

    add("aggregate", "What is the average current salary in the Development department?",
        stats(dept_no=dm["Development"])[0]["avg_salary"], {"dept_no": dm["Development"]})
    add("aggregate", "What is the maximum current salary in the Sales department?",
        stats(dept_no=dm["Sales"])[0]["max_salary"], {"dept_no": dm["Sales"]})
    add("aggregate", "How many employees currently hold a salary record (are on the active "
        "payroll)?", stats()[0]["count"], {})
    add("aggregate", "How many employees in the Sales department currently earn at least "
        "150000?", _cnt(dept_no=dm["Sales"], min_salary=150000), {"dept_no": dm["Sales"]})

    r0 = raised[0]
    d_r0 = r0["sal"][1]["from_date"]
    add("temporal", f"What was the annual salary of {full(r0['row'])} on {d_r0}?",
        r0["sal"][1]["salary"], {"emp_no": r0["row"]["emp_no"], "at_date": d_r0})
    add("temporal", f"How many salary records has {full(r0['row'])} had in total over their "
        f"employment history?", len(r0["sal"]), {"emp_no": r0["row"]["emp_no"]})
    p0 = promoted[0]
    d_p0 = p0["tit"][1]["from_date"]
    add("temporal", f"What job title did {full(p0['row'])} hold on {d_p0}?",
        p0["tit"][1]["title"], {"emp_no": p0["row"]["emp_no"], "at_date": d_p0})
    t0 = transferred[0]
    d_t0 = t0["dep"][1]["from_date"]
    add("temporal", f"In which department did {full(t0['row'])} work on {d_t0}? Give the "
        f"department name.", t0["dep"][1]["dept_name"],
        {"emp_no": t0["row"]["emp_no"], "at_date": d_t0})

    top_all = top(n=1)[0]
    add("superlative", "Who is the top earner in the entire company? Give the full name.",
        full(top_all), {"emp_no": top_all["emp_no"]})
    top_hr = top(dept_no=dm["Human Resources"], n=1)[0]
    add("superlative", "Who is the top earner in the Human Resources department? Give the "
        "full name.", full(top_hr), {"emp_no": top_hr["emp_no"]})

    # ── B: boundary traps (16) ───────────────────────────────────────────
    # every trap asserts the boundary is populated: the naive inclusive
    # (or year-overflow) filter provably returns a different count
    def add_strict_before(scope, label, thr, **flt):
        strict = _cnt(hired_before=_day_shift(thr, -1), **flt)
        naive = _cnt(hired_before=thr, **flt)
        assert naive > strict, f"no boundary for hire-before {thr} {label}"
        add("boundary", f"How many employees {label} were hired strictly before {thr}?",
            strict, {**flt, "hired_before": _day_shift(thr, -1)})

    def add_strict_after(scope, label, thr, **flt):
        strict = _cnt(hired_after=_day_shift(thr, 1), **flt)
        naive = _cnt(hired_after=thr, **flt)
        assert naive > strict, f"no boundary for hire-after {thr} {label}"
        add("boundary", f"How many employees {label} were hired strictly after {thr}?",
            strict, {**flt, "hired_after": _day_shift(thr, 1)})

    add_strict_before(None, "company-wide", "1990-01-01")
    for thr in ("1996-01-01", "1995-06-01", "1992-01-01", "1998-01-01"):
        try:
            add_strict_after(None, "in the Production department", thr,
                             dept_no=dm["Production"])
            break
        except AssertionError:
            continue
    else:
        raise AssertionError("no strict-after boundary in Production")
    for lo, hi_ in (("1990-01-01", "1992-01-01"), ("1992-01-01", "1994-01-01"),
                    ("1988-06-01", "1991-06-01")):
        strict = _cnt(dept_no=dm["Sales"], hired_after=_day_shift(lo, 1),
                      hired_before=_day_shift(hi_, -1))
        naive = _cnt(dept_no=dm["Sales"], hired_after=lo, hired_before=hi_)
        if naive > strict:
            add("boundary", f"How many employees in the Sales department were hired strictly "
                f"between {lo} and {hi_} (both ends exclusive)?", strict,
                {"dept_no": dm["Sales"], "hired_after": _day_shift(lo, 1),
                 "hired_before": _day_shift(hi_, -1)})
            break
    else:
        raise AssertionError("no strict-between hire boundary in Sales")
    for thr in ("1992-06-15", "1993-03-10", "1995-09-20", "1997-04-01"):
        try:
            add_strict_before(None, "in the Finance department", thr, dept_no=dm["Finance"])
            break
        except AssertionError:
            continue
    else:
        raise AssertionError("no mid-date hire boundary in Finance")

    # period-inclusive: the year-overflow naive filter differs
    for year in (1985, 1986, 1987, 1988):
        correct = _cnt(hired_after=f"{year}-01-01", hired_before=f"{year}-12-31")
        naive = _cnt(hired_after=f"{year}-01-01", hired_before=f"{year + 1}-01-01")
        if naive > correct:
            add("boundary", f"How many employees were hired during {year} (inclusive: "
                f"{year}-01-01 through {year}-12-31)?", correct,
                {"hired_after": f"{year}-01-01", "hired_before": f"{year}-12-31"})
            break
    else:
        raise AssertionError("no year-boundary hires")
    for dn in ("Marketing", "Human Resources", "Customer Service"):
        correct = _cnt(dept_no=dm[dn], hired_after="1980-01-01", hired_before="1989-12-31")
        naive = _cnt(dept_no=dm[dn], hired_after="1980-01-01", hired_before="1990-01-01")
        if naive > correct:
            add("boundary", f"How many employees in the {dn} department were hired during "
                f"the 1980s (inclusive: 1980-01-01 through 1989-12-31)?", correct,
                {"dept_no": dm[dn], "hired_before": "1989-12-31"})
            break
    else:
        raise AssertionError("no decade-boundary hires")

    # band complements: not expressible as a single filter
    for dn, lo, hi_ in (("Production", 40000, 50000), ("Development", 60000, 70000)):
        total = _cnt(dept_no=dm[dn])
        band = _cnt(dept_no=dm[dn], min_salary=lo, max_salary=hi_)
        assert band > 0 and total - band > 0
        add("boundary", f"How many employees in the {dn} department currently do NOT earn "
            f"between {lo} and {hi_} (inclusive)?", total - band,
            {"dept_no": dm[dn], "band": [lo, hi_]})

    # mixed inclusive/exclusive salary bounds
    for dn, lo, hi_ in (("Marketing", 80000, 100000), ("Customer Service", 90000, 110000)):
        strict = _cnt(dept_no=dm[dn], min_salary=lo, max_salary=hi_ - 1)
        naive = _cnt(dept_no=dm[dn], min_salary=lo, max_salary=hi_)
        if naive > strict:
            add("boundary", f"How many employees in the {dn} department currently earn at "
                f"least {lo} but strictly less than {hi_}?", strict,
                {"dept_no": dm[dn], "min_salary": lo, "max_salary": hi_ - 1})
            break
    else:
        raise AssertionError("no mixed-boundary salary trap")
    for dn, lo, hi_ in (("Sales", 60000, 80000), ("Finance", 50000, 70000)):
        strict = _cnt(dept_no=dm[dn], min_salary=lo + 1, max_salary=hi_)
        naive = _cnt(dept_no=dm[dn], min_salary=lo, max_salary=hi_)
        if naive > strict:
            add("boundary", f"How many employees in the {dn} department currently earn more "
                f"than {lo} but at most {hi_}?", strict,
                {"dept_no": dm[dn], "min_salary": lo + 1, "max_salary": hi_})
            break
    else:
        raise AssertionError("no mixed-boundary-2 salary trap")
    for thr in (120000, 140000, 100000):
        naive = _cnt(dept_no=dm["Sales"], min_salary=thr)
        strict = _cnt(dept_no=dm["Sales"], min_salary=thr + 1)
        if naive > strict:
            add("boundary", f"How many employees in the Sales department currently earn "
                f"strictly more than {thr}?", strict,
                {"dept_no": dm["Sales"], "min_salary": thr + 1})
            break
    else:
        raise AssertionError("no strict-lower boundary in Sales")

    # salary strict variants
    for thr in (60000, 40000, 50000, 45000, 55000, 65000, 70000):
        naive = _cnt(dept_no=dm["Production"], max_salary=thr)
        strict = _cnt(dept_no=dm["Production"], max_salary=thr - 1)
        if naive > strict:
            add("boundary", f"How many employees in the Production department currently "
                f"earn strictly less than {thr}?", strict,
                {"dept_no": dm["Production"], "max_salary": thr - 1})
            break
    else:
        raise AssertionError("no strict-upper boundary in Production")
    for lo, hi_ in ((60000, 80000), (50000, 70000), (40000, 60000), (55000, 75000)):
        naive = _cnt(dept_no=dm["Development"], min_salary=lo, max_salary=hi_)
        strict = _cnt(dept_no=dm["Development"], min_salary=lo + 1, max_salary=hi_ - 1)
        if naive > strict:
            add("boundary", f"How many employees in the Development department currently "
                f"earn strictly between {lo} and {hi_} (both ends exclusive)?", strict,
                {"dept_no": dm["Development"], "min_salary": lo + 1, "max_salary": hi_ - 1})
            break
    else:
        raise AssertionError("no strict-between salary boundary in Development")
    for thr in (100000, 90000, 80000, 120000, 110000, 150000):
        naive = _cnt(dept_no=dm["Finance"], min_salary=thr)
        strict = _cnt(dept_no=dm["Finance"], min_salary=thr + 1)
        if naive > strict:
            add("boundary", f"How many employees in the Finance department currently earn "
                f"strictly more than {thr}?", strict,
                {"dept_no": dm["Finance"], "min_salary": thr + 1})
            break
    else:
        raise AssertionError("no strict-lower boundary in Finance")

    # decade complement (the proven biter), fresh department
    for dn, decade_lo, decade_hi in (("Sales", "1990-01-01", "1999-12-31"),
                                     ("Marketing", "1985-01-01", "1989-12-31")):
        total = _cnt(dept_no=dm[dn])
        in_dec = _cnt(dept_no=dm[dn], hired_after=decade_lo, hired_before=decade_hi)
        assert 0 < in_dec < total
        add("boundary", f"How many employees in the {dn} department were NOT hired during "
            f"{decade_lo[:4]}–{decade_hi[:4]} ({decade_lo} through {decade_hi})?",
            total - in_dec, {"dept_no": dm[dn], "decade": [decade_lo, decade_hi]})

    # ── C: exclusion counting (11) ───────────────────────────────────────
    dept_totals = {dn: _cnt(dept_no=dm[dn]) for dn in names}
    combos = []
    for dn in names:
        for t in TITLES:
            if len(combos) >= 4:
                break
            filtered = _cnt(dept_no=dm[dn], title=t)
            if filtered >= 50 and dept_totals[dn] - filtered >= 100 \
                    and all(c[0] != dn for c in combos):
                combos.append((dn, t, filtered))
    assert len(combos) == 4, f"only {len(combos)} dept-title exclusion combos"
    for dn, t, filtered in combos:
        add("exclusion", f"How many employees currently work in the {dn} department and do "
            f"NOT hold the title {t}?", dept_totals[dn] - filtered,
            {"dept_no": dm[dn], "title": t})

    # senior-title complement: concept -> TWO exact titles -> two searches + complement
    for dn in ("Sales", "Research"):
        total = _cnt(dept_no=dm[dn])
        se = _cnt(dept_no=dm[dn], title="Senior Engineer")
        ss = _cnt(dept_no=dm[dn], title="Senior Staff")
        assert total - se - ss > 0
        add("exclusion", f"How many employees in the {dn} department do NOT hold a senior "
            f"title (Senior Engineer or Senior Staff)?", total - se - ss,
            {"dept_no": dm[dn], "n_senior": se + ss})

    # decade complements, exclusion-style (total minus period)
    for dn, lo, hi_ in (("Development", "1990-01-01", "1999-12-31"),
                        ("Production", "1990-01-01", "1999-12-31")):
        in_dec = _cnt(dept_no=dm[dn], hired_after=lo, hired_before=hi_)
        assert 0 < in_dec < dept_totals[dn]
        add("exclusion", f"How many employees in the {dn} department were NOT hired during "
            f"the 1990s ({lo} through {hi_})?", dept_totals[dn] - in_dec,
            {"dept_no": dm[dn], "decade": [lo, hi_]})

    eng_total = _cnt(title="Engineer")
    eng_mkt = _cnt(title="Engineer", dept_no=dm["Marketing"])
    assert eng_total >= 100 and eng_total - eng_mkt >= 1
    add("exclusion", "How many employees with the title Engineer currently work outside the "
        "Marketing department?", eng_total - eng_mkt,
        {"title": "Engineer", "dept_no": dm["Marketing"]})

    prod_f = _cnt(dept_no=dm["Production"], gender="F")
    prod_f_eng = _cnt(dept_no=dm["Production"], gender="F", title="Engineer")
    assert prod_f - prod_f_eng > 0
    add("exclusion", "How many female employees in the Production department do NOT hold "
        "the title Engineer?", prod_f - prod_f_eng,
        {"dept_no": dm["Production"], "gender": "F", "title": "Engineer"})

    prod_under = _cnt(dept_no=dm["Production"], max_salary=59999)
    prod_under_eng = _cnt(dept_no=dm["Production"], max_salary=59999, title="Engineer")
    assert prod_under - prod_under_eng > 0
    add("exclusion", "How many employees in the Production department currently earn less "
        "than 60000 and do NOT hold the title Engineer?", prod_under - prod_under_eng,
        {"dept_no": dm["Production"], "title": "Engineer", "max_salary": 59999})

    # ── D: ambiguous-name disambiguation (9) ─────────────────────────────
    amb = find_ambiguous(pool, need=14)
    assert len(amb) >= 10, f"only {len(amb)} disambiguation candidates"

    # stated-qualifier floor (3): salary, title, hire_date
    stated_done = {"salary": 0, "title": 0, "hire_date": 0}
    for cand in amb:
        if all(v >= 1 for v in stated_done.values()):
            break
        quals = cand["quals"]
        ask = None
        qual = None
        for want in ("salary", "hire_date", "title"):
            if stated_done[want] >= 1:
                continue
            for kind in (["dept", "gender"] if want == "title" else ["dept", "title", "gender"]):
                for q in quals:
                    if q["kind"] == kind:
                        ask, qual = want, q
                        break
                if ask:
                    break
            if ask:
                break
        if ask is None:
            continue
        stated_done[ask] += 1
        detail = emp_detail(cand["row"]["emp_no"])
        fname, lname = detail["first_name"], detail["last_name"]
        stem = f"There are multiple employees named {fname} {lname}. What is the "
        if ask == "salary":
            add("disambiguation", stem + f"current annual salary of the one {qual['phrase']}?",
                detail["salary"], {"emp_no": detail["emp_no"]})
        elif ask == "title":
            add("disambiguation", stem + f"current job title of the one {qual['phrase']}?",
                detail["title"]["title"], {"emp_no": detail["emp_no"]})
        else:
            add("disambiguation", stem + f"hire date of the one {qual['phrase']}? Answer in "
                f"YYYY-MM-DD format.", detail["hire_date"], {"emp_no": detail["emp_no"]})
    assert all(v >= 1 for v in stated_done.values()), stated_done

    # same-department-as-colleague (3): qualifier references another entity
    same_dept = []
    for cand in amb:
        if len(same_dept) >= 3:
            break
        t = cand["row"]
        d = t["dept_no"]
        if search(first_name=t["first_name"], last_name=t["last_name"], dept_no=d,
                  page_size=1)["total"] != 1:
            continue
        y = next((it for it in search(dept_no=d, page_size=100, sort="emp_no")["items"]
                  if it["emp_no"] != t["emp_no"] and unique_named(it)), None)
        if y is None:
            continue
        same_dept.append((t, y))
    assert len(same_dept) == 3, "no same-department-as-colleague candidates"
    for (t, y), ask in zip(same_dept, ("salary", "title", "salary")):
        detail = emp_detail(t["emp_no"])
        fname, lname = detail["first_name"], detail["last_name"]
        if ask == "salary":
            add("disambiguation", f"There are multiple employees named {fname} {lname}. What "
                f"is the current annual salary of the one who works in the same department "
                f"as {full(y)}?", detail["salary"],
                {"emp_no": detail["emp_no"], "colleague": y["emp_no"]})
        else:
            add("disambiguation", f"There are multiple employees named {fname} {lname}. What "
                f"is the current job title of the one who works in the same department as "
                f"{full(y)}?", detail["title"]["title"],
                {"emp_no": detail["emp_no"], "colleague": y["emp_no"]})

    # double disambiguation (3): ambiguous name qualified by the department
    # of ANOTHER ambiguous name resolved by a stated qualifier; distinct
    # target members and (anchor, qualifier) combos across all three
    doubles = []
    used_members: set[int] = set()
    seen_anchor_qual: set[tuple[int, str]] = set()
    for pass_no in (1, 2):
        for cand_b in amb:
            if len(doubles) >= 3:
                break
            b = cand_b["row"]
            for q in cand_b["quals"]:
                if (b["emp_no"], q["kind"]) in seen_anchor_qual:
                    continue
                hit = None
                for cand_a in amb:
                    ar = cand_a["row"]
                    if ar["emp_no"] == b["emp_no"]:
                        continue
                    h = search(first_name=ar["first_name"], last_name=ar["last_name"],
                               dept_no=b["dept_no"], page_size=1)
                    if h["total"] == 1 and h["items"][0]["emp_no"] != b["emp_no"] \
                            and h["items"][0]["emp_no"] not in used_members:
                        hit = h["items"][0]
                        break
                if hit is None:
                    continue
                doubles.append((hit, b, q))
                used_members.add(hit["emp_no"])
                seen_anchor_qual.add((b["emp_no"], q["kind"]))
                break
        if len(doubles) >= 3:
            break
    assert len(doubles) == 3, f"only {len(doubles)} double-disambiguation configs"
    for member, b, q in doubles:
        detail = emp_detail(member["emp_no"])
        add("disambiguation", f"There are multiple employees named {detail['first_name']} "
            f"{detail['last_name']}. What is the current job title of the one who works in "
            f"the same department as the {b['first_name']} {b['last_name']} {q['phrase']}?",
            detail["title"]["title"], {"emp_no": detail["emp_no"], "anchor": b["emp_no"]})

    # ── E: cross-entity comparisons (8) ──────────────────────────────────
    dept_avg: dict[str, float] = {}
    far_above = near_above = None
    for row in pool[10:]:
        dn = row["dept_no"]
        if dn not in dept_avg:
            dept_avg[dn] = stats(dept_no=dn)[0]["avg_salary"]
        if not row["salary"]:
            continue
        rel = (row["salary"] - dept_avg[dn]) / dept_avg[dn]
        if rel >= 0.30 and far_above is None:
            far_above = row
        elif 0.05 <= rel <= 0.15 and near_above is None:
            near_above = row
        if far_above and near_above:
            break
    assert far_above and near_above, "no anchors in the required vs-average bands"
    add("comparison", f"Does {full(far_above)} currently earn at least 20% more than the "
        f"average salary of their department? Answer true or false.", "true",
        {"emp_no": far_above["emp_no"], "dept_no": far_above["dept_no"]})
    add("comparison", f"Does {full(near_above)} currently earn at least 20% more than the "
        f"average salary of their department? Answer true or false.", "false",
        {"emp_no": near_above["emp_no"], "dept_no": near_above["dept_no"]})

    mgr_pairs = []
    mgr_used: set[str] = set()
    for i, dn1 in enumerate(names):
        for dn2 in names[i + 1:]:
            if dn1 in mgr_used or dn2 in mgr_used:
                continue
            s1, s2 = mgr_sal[dn1], mgr_sal[dn2]
            if not s1 or not s2:
                continue
            if abs(s1 - s2) / max(s1, s2) <= 0.15:
                mgr_pairs.append((dn1, dn2))
                mgr_used.update({dn1, dn2})
        if len(mgr_pairs) >= 2:
            break
    assert len(mgr_pairs) == 2, f"only {len(mgr_pairs)} near-miss manager pairs"
    dn1, dn2 = mgr_pairs[0]
    hi, lo = (dn1, dn2) if mgr_sal[dn1] >= mgr_sal[dn2] else (dn2, dn1)
    add("comparison", f"Does the current manager of the {hi} department currently earn more "
        f"than the current manager of the {lo} department? Answer true or false.", "true",
        {"mgr_hi": mgr_info[hi]["emp_no"], "mgr_lo": mgr_info[lo]["emp_no"]})
    dn1, dn2 = mgr_pairs[1]
    hi, lo = (dn1, dn2) if mgr_sal[dn1] >= mgr_sal[dn2] else (dn2, dn1)
    add("comparison", f"Does the current manager of the {lo} department currently earn more "
        f"than the current manager of the {hi} department? Answer true or false.", "false",
        {"mgr_hi": mgr_info[hi]["emp_no"], "mgr_lo": mgr_info[lo]["emp_no"]})

    # strict-date booleans, near-miss anchors on both sides of the boundary
    for thr in ("1990-01-01", "1992-01-01", "1988-01-01"):
        before = next((r for r in pool[10:] if r["hire_date"] < thr
                       and (dt.date.fromisoformat(thr)
                            - dt.date.fromisoformat(r["hire_date"])).days <= 150), None)
        after = next((r for r in pool[10:] if r["hire_date"] > thr
                      and (dt.date.fromisoformat(r["hire_date"])
                           - dt.date.fromisoformat(thr)).days <= 150), None)
        if before and after:
            add("comparison", f"Was {full(before)} hired strictly before {thr}? Answer true "
                f"or false.", "true", {"emp_no": before["emp_no"]})
            add("comparison", f"Was {full(after)} hired strictly before {thr}? Answer true "
                f"or false.", "false", {"emp_no": after["emp_no"]})
            break
    else:
        raise AssertionError("no strict-date comparison anchors")

    member = outsider = None
    for dn in names:
        top10 = top(dept_no=dm[dn], n=10)
        for pos in range(9, 4, -1):
            if pos < len(top10) and unique_named(top10[pos]):
                member = top10[pos]
                member_dept = dn
                break
        if member:
            break
    for row in pool[10:]:
        top10 = top(dept_no=row["dept_no"], n=10)
        if row["emp_no"] not in {m["emp_no"] for m in top10}:
            outsider = row
            break
    assert member and outsider, "no top-10 membership pair found"
    add("comparison", f"Is {full(member)} currently among the 10 highest earners of the "
        f"{member_dept} department? Answer true or false.", "true",
        {"emp_no": member["emp_no"], "dept_no": dm[member_dept]})
    add("comparison", f"Is {full(outsider)} currently among the 10 highest earners of the "
        f"{outsider['dept_name']} department? Answer true or false.", "false",
        {"emp_no": outsider["emp_no"], "dept_no": outsider["dept_no"]})

    # ── F: arithmetic over retrievals (6) ────────────────────────────────
    used: set[int] = set()
    diff_pair = None
    for i in range(10, len(pool)):
        for j in range(i + 1, len(pool)):
            x, y = pool[i], pool[j]
            if x["emp_no"] in used or y["emp_no"] in used or not x["salary"] or not y["salary"]:
                continue
            if abs(x["salary"] - y["salary"]) / max(x["salary"], y["salary"]) >= 0.05:
                diff_pair = (x, y)
                break
        if diff_pair:
            break
    assert diff_pair, "no salary-diff pair"
    x, y = diff_pair
    hi, lo = (x, y) if x["salary"] >= y["salary"] else (y, x)
    add("arithmetic", f"How many dollars more does {full(hi)} currently earn per year than "
        f"{full(lo)}?", hi["salary"] - lo["salary"],
        {"emp_hi": hi["emp_no"], "emp_lo": lo["emp_no"]})

    pct_row = next(r for r in pool[30:]
                   if r["salary"] and (r["salary"] - dept_avg.setdefault(
                       r["dept_no"], stats(dept_no=r["dept_no"])[0]["avg_salary"]))
                   / dept_avg[r["dept_no"]] > 0.05)
    pct = round((pct_row["salary"] - dept_avg[pct_row["dept_no"]])
                / dept_avg[pct_row["dept_no"]] * 100, 2)
    add("arithmetic", f"By what percentage does {full(pct_row)}'s current salary exceed the "
        f"average salary of their department? Give the percentage rounded to 2 decimals "
        f"(e.g. 12.34).", pct, {"emp_no": pct_row["emp_no"], "dept_no": pct_row["dept_no"]})

    hc_by_name = {r["dept_name"]: r["headcount"]
                  for r in api("GET", "/api/stats/headcount")}
    add("arithmetic", "How many times more employees currently work in the Development "
        "department than in the Finance department? Give the ratio rounded to 2 decimals.",
        round(hc_by_name["Development"] / hc_by_name["Finance"], 2),
        {"dept_hi": dm["Development"], "dept_lo": dm["Finance"]})

    rr = raised[2]
    old = rr["sal"][0]
    current = emp_detail(rr["row"]["emp_no"])["salary"]
    assert current > old["salary"], "expected salary growth"
    add("arithmetic", f"How many dollars more does {full(rr['row'])} earn now than on "
        f"{old['from_date']}?", current - old["salary"], {"emp_no": rr["row"]["emp_no"]})

    for dn, thr in (("Sales", 100000), ("Production", 60000)):
        cnt = _cnt(dept_no=dm[dn], max_salary=thr - 1)
        tot = _cnt(dept_no=dm[dn])
        assert cnt > 0
        add("arithmetic", f"What percentage of the {dn} department's employees currently "
            f"earn strictly less than {thr}? Give the percentage rounded to 2 decimals "
            f"(e.g. 12.34).", round(cnt / tot * 100, 2), {"dept_no": dm[dn]})

    # ── G: hardened verification (7) ─────────────────────────────────────
    p = promoted[0]
    d_old = p["tit"][0]["from_date"]
    add("verification", f"Did {full(p['row'])} hold the title {p['tit'][0]['title']} on "
        f"{d_old}? Answer true or false.", "true",
        {"emp_no": p["row"]["emp_no"], "at_date": d_old})
    add("verification", f"Did {full(p['row'])} hold the title {p['tit'][1]['title']} on "
        f"{d_old}? Answer true or false.", "false",
        {"emp_no": p["row"]["emp_no"], "at_date": d_old})

    ten_true = ten_false = None
    for thr in ("1995-01-01", "1990-01-01", "2000-01-01", "2005-01-01", "2010-01-01"):
        t_before = next((dn for dn in names if mgr_info[dn]["from_date"] < thr), None)
        t_after = next((dn for dn in names if mgr_info[dn]["from_date"] > thr), None)
        if ten_true is None and t_before:
            ten_true = (t_before, thr)
        if ten_false is None and t_after:
            ten_false = (t_after, thr)
        if ten_true and ten_false:
            break
    assert ten_true and ten_false, "no manager-tenure pair"
    add("verification", f"Has the current manager of the {ten_true[0]} department been "
        f"managing it since before {ten_true[1]}? Answer true or false.", "true",
        {"manager_emp_no": mgr_info[ten_true[0]]["emp_no"]})
    add("verification", f"Has the current manager of the {ten_false[0]} department been "
        f"managing it since before {ten_false[1]}? Answer true or false.", "false",
        {"manager_emp_no": mgr_info[ten_false[0]]["emp_no"]})

    dept_no_to_name = {dm[dn]: dn for dn in names}
    before = after = None
    for row in pool[10:]:
        dn = dept_no_to_name[row["dept_no"]]
        if mgr_info[dn]["emp_no"] == row["emp_no"]:
            continue
        mgr_hd = emp_detail(mgr_info[dn]["emp_no"])["hire_date"]
        if row["hire_date"] < mgr_hd and before is None:
            before = (row, dn)
        elif row["hire_date"] > mgr_hd and after is None:
            after = (row, dn)
        if before and after:
            break
    assert before and after, "no joined-before/after-manager anchor pairs"
    row, dn = before
    add("verification", f"Did {full(row)} join the company before the current manager of "
        f"their department did? Answer true or false.", "true",
        {"emp_no": row["emp_no"], "manager_emp_no": mgr_info[dn]["emp_no"]})
    row, dn = after
    add("verification", f"Did {full(row)} join the company before the current manager of "
        f"their department did? Answer true or false.", "false",
        {"emp_no": row["emp_no"], "manager_emp_no": mgr_info[dn]["emp_no"]})

    d_in = t0["dep"][0]["from_date"]
    add("verification", f"Did {full(t0['row'])} work in the {t0['dep'][0]['dept_name']} "
        f"department on {d_in}? Answer true or false.", "true",
        {"emp_no": t0["row"]["emp_no"], "at_date": d_in})

    # ── H: deep multi-hop chains (6) ─────────────────────────────────────
    anchor = pool[20]
    dn20 = anchor["dept_no"]
    te20 = top(dept_no=dn20, n=1)[0]
    add("deep", f"What is the current annual salary of the top earner in the department "
        f"where {full(anchor)} works?", te20["salary"],
        {"emp_no": anchor["emp_no"], "dept_no": dn20, "top_emp_no": te20["emp_no"]})

    chain = None
    for cand in pool[21:30]:
        dep = dept_history(cand["emp_no"])
        for at in ("1998-06-01", "2000-06-01", "2005-06-01"):
            row = next((r for r in dep if covers(r, at)), None)
            if row is None:
                continue
            mgrs = api("GET", f"/api/departments/{row['dept_no']}/managers",
                       params={"at_date": at})
            if mgrs:
                chain = (cand, at, mgrs[0])
                break
        if chain:
            break
    assert chain, "no dept->manager at-date chain found"
    cand, at, mgr_rec = chain
    add("deep", f"Who was the manager of the department where {full(cand)} worked on {at}? "
        f"Give the manager's full name.", full(mgr_rec),
        {"emp_no": cand["emp_no"], "at_date": at, "manager_emp_no": mgr_rec["emp_no"]})

    fin_top = top(dept_no=dm["Finance"], n=1)[0]
    ln = fin_top["last_name"]
    add("deep", "How many employees currently share the last name of the top earner in the "
        "Finance department?", _cnt(last_name=ln), {"top_emp_no": fin_top["emp_no"]})

    sales_top = top(dept_no=dm["Sales"], n=1)[0]
    add("deep", "Is the current manager of the Sales department its own top earner? Answer "
        "true or false.", str(mgr_info["Sales"]["emp_no"] == sales_top["emp_no"]).lower(),
        {"manager_emp_no": mgr_info["Sales"]["emp_no"], "top_emp_no": sales_top["emp_no"]})

    cs_top10 = top(dept_no=dm["Customer Service"], n=10)
    add("deep", "Is the current manager of the Customer Service department among that "
        "department's 10 highest earners? Answer true or false.",
        str(mgr_info["Customer Service"]["emp_no"] in {m["emp_no"] for m in cs_top10}).lower(),
        {"manager_emp_no": mgr_info["Customer Service"]["emp_no"]})

    fin2 = top(dept_no=dm["Finance"], n=2)[1]
    fin2_dept_no = next(r["dept_no"] for r in api("GET", "/api/departments")
                        if r["dept_name"] == fin2["dept_name"])
    fin2_mgr = api("GET", f"/api/departments/{fin2_dept_no}")["manager"]
    add("deep", "What is the current annual salary of the manager of the department where "
        "the second-highest earner of the Finance department works?",
        emp_detail(fin2_mgr["emp_no"])["salary"],
        {"top2_emp_no": fin2["emp_no"], "dept_no": fin2_dept_no,
         "manager_emp_no": fin2_mgr["emp_no"]})

    # ── I: ranked positions & top-n set logic (6) ────────────────────────
    fin_top3 = top(dept_no=dm["Finance"], n=3)
    assert len(fin_top3) >= 3
    add("rank", "Who is the second-highest earner in the Finance department? Give the full "
        "name.", full(fin_top3[1]), {"emp_no": fin_top3[1]["emp_no"]})
    sales_top3 = top(dept_no=dm["Sales"], n=3)
    assert len(sales_top3) >= 3
    add("rank", "What is the current annual salary of the third-highest earner in the Sales "
        "department?", sales_top3[2]["salary"], {"emp_no": sales_top3[2]["emp_no"]})
    dev_top5 = top(dept_no=dm["Development"], n=5)
    assert len(dev_top5) >= 5
    add("rank", "What is the current annual salary of the fifth-highest earner in the "
        "Development department?", dev_top5[4]["salary"], {"emp_no": dev_top5[4]["emp_no"]})

    ranked_ex = []
    for dn in names:
        if len(ranked_ex) >= 2:
            break
        rows5 = top(dept_no=dm[dn], n=5)
        n_senior = sum(1 for m in rows5
                       if m["title"] in ("Senior Engineer", "Senior Staff"))
        if 1 <= n_senior <= 4:
            ranked_ex.append((dn, n_senior))
    assert len(ranked_ex) == 2, f"only {len(ranked_ex)} ranked-exclusion departments"
    for dn, n_senior in ranked_ex:
        add("rank", f"How many of the 5 highest earners in the {dn} department do NOT hold "
            f"a senior title (Senior Engineer or Senior Staff)?", 5 - n_senior,
            {"dept_no": dm[dn], "n_senior": n_senior})

    dev_top10 = top(dept_no=dm["Development"], n=10)
    assert len(dev_top10) >= 10
    n_three = sum(1 for m in dev_top10
                  if m["title"] in ("Senior Engineer", "Senior Staff", "Technique Leader"))
    assert 1 <= n_three <= 9, "degenerate 3-way set count"
    add("rank", "How many of the 10 highest earners in the Development department hold one "
        "of the titles Senior Engineer, Senior Staff, or Technique Leader?", n_three,
        {"dept_no": dm["Development"], "n_three": n_three})

    # ── J: fan-out superlatives (3) ──────────────────────────────────────
    by_dept = stats(group_by="dept")
    assert len(by_dept) == len(names)
    hi = max(by_dept, key=lambda r: r["avg_salary"])
    add("fanout", "Which department has the highest average current salary? Give the "
        "department name.", hi["dept_name"], {"dept_no": hi["dept_no"]})
    second = sorted(by_dept, key=lambda r: r["avg_salary"], reverse=True)[1]
    add("fanout", "Which department has the second-highest average current salary? Give the "
        "department name.", second["dept_name"], {"dept_no": second["dept_no"]})
    hc_rows = api("GET", "/api/stats/headcount")
    fewest = min(hc_rows, key=lambda r: r["headcount"])
    add("fanout", "Which department currently has the fewest employees? Give the department "
        "name.", fewest["dept_name"], {"dept_no": fewest["dept_no"]})

    # ── K: nested derived-threshold chains (4) ───────────────────────────
    dev_mgr_sal = mgr_sal["Development"]
    n_ge = _cnt(dept_no=dm["Sales"], min_salary=dev_mgr_sal)
    assert n_ge > 0
    add("nested", "How many employees in the Sales department currently earn at least as "
        "much as the current manager of the Development department?", n_ge,
        {"min_salary": dev_mgr_sal, "dept_no": dm["Sales"]})

    mkt_top = top(dept_no=dm["Marketing"], n=1)[0]
    s_top = mkt_top["salary"]
    tie = _cnt(dept_no=dm["Marketing"], min_salary=s_top, max_salary=s_top)
    assert tie == 1, "Marketing top earner salary is not unique"
    n_below = _cnt(dept_no=dm["Marketing"], max_salary=s_top - 1)
    add("nested", "How many employees in the Marketing department currently earn strictly "
        "less than the Marketing department's own top earner?", n_below,
        {"dept_no": dm["Marketing"], "max_salary": s_top - 1})

    mgr_on_mgr = None
    for d_hi, d_lo in (("Human Resources", "Finance"), ("Sales", "Development"),
                       ("Finance", "Human Resources"), ("Marketing", "Sales")):
        start = mgr_info[d_lo]["from_date"]
        sal = salary_at(mgr_info[d_hi]["emp_no"], start)
        if sal is not None:
            mgr_on_mgr = (d_hi, d_lo, start, sal)
            break
    assert mgr_on_mgr, "no manager-on-manager-date chain"
    d_hi, d_lo, start, sal = mgr_on_mgr
    add("nested", f"What was the annual salary of the current manager of the {d_hi} "
        f"department on the date the current manager of the {d_lo} department started "
        f"managing the {d_lo} department?", sal,
        {"manager_emp_no": mgr_info[d_hi]["emp_no"], "at_date": start})

    fin_start = mgr_info["Finance"]["from_date"]
    at_start = None
    for cand in list(transferred) + pool[20:40]:
        if "dep" in cand:
            row_obj, dep = cand["row"], cand["dep"]
        else:
            row_obj, dep = cand, dept_history(cand["emp_no"])
        rec = next((r for r in dep if covers(r, fin_start)), None)
        if rec is not None:
            at_start = (row_obj, rec)
            break
    assert at_start, "no anchor covering Finance-manager start date"
    row_obj, rec = at_start
    add("nested", f"In which department did {full(row_obj)} work on the date the current "
        f"manager of the Finance department started managing it? Give the department name.",
        rec["dept_name"], {"emp_no": row_obj["emp_no"], "at_date": fin_start})

    # ── L: history indexing (3) ──────────────────────────────────────────
    r2 = raised[2]
    assert len(r2["sal"]) >= 2
    add("history", f"What was the annual salary of {full(r2['row'])} immediately before "
        f"their most recent salary change?", r2["sal"][-2]["salary"],
        {"emp_no": r2["row"]["emp_no"]})
    p2 = promoted[2]
    assert p2["tit"][-2]["title"] != p2["tit"][-1]["title"]
    add("history", f"What job title did {full(p2['row'])} hold immediately before their "
        f"current one?", p2["tit"][-2]["title"], {"emp_no": p2["row"]["emp_no"]})

    raise_band = None
    for row in pool[:60]:
        sal = salary_history(row["emp_no"])
        if len(sal) < 2:
            continue
        pct = (sal[-1]["salary"] - sal[-2]["salary"]) / sal[-2]["salary"] * 100
        if 5 < pct < 50:
            raise_band = row
            break
    assert raise_band, "no raise-band anchor"
    add("history", f"Did {full(raise_band)}'s most recent salary change increase their "
        f"salary by more than 5%? Answer true or false.", "true",
        {"emp_no": raise_band["emp_no"]})

    # ── M: cross-population comparisons (2) ──────────────────────────────
    cc_combos = [("Development", "Engineer"), ("Production", "Engineer"),
                 ("Sales", "Senior Staff"), ("Marketing", "Senior Staff"),
                 ("Finance", "Staff"), ("Human Resources", "Staff")]
    cc_vals = {(dn, t): stats(dept_no=dm[dn], title=t)[0]["avg_salary"]
               for dn, t in cc_combos}
    cc_near = []
    cc_keys = list(cc_vals)
    for i in range(len(cc_keys)):
        for j in range(i + 1, len(cc_keys)):
            va, vb = cc_vals[cc_keys[i]], cc_vals[cc_keys[j]]
            if abs(va - vb) / max(va, vb) <= 0.12:
                cc_near.append((cc_keys[i], cc_keys[j]))
    assert cc_near, "no near-miss crosscompare pairs"
    (da, ta), (db, tb) = cc_near[0]
    hi_side, lo_side = ((da, ta), (db, tb)) \
        if cc_vals[(da, ta)] >= cc_vals[(db, tb)] else ((db, tb), (da, ta))
    add("crosscompare", f"Is the average current salary of employees with the title "
        f"{hi_side[1]} in the {hi_side[0]} department higher than that of employees with "
        f"the title {lo_side[1]} in the {lo_side[0]} department? Answer true or false.",
        "true", {"dept_hi": dm[hi_side[0]], "dept_lo": dm[lo_side[0]]})
    if len(cc_near) > 1:
        (da, ta), (db, tb) = cc_near[1]
    hi_side, lo_side = ((da, ta), (db, tb)) \
        if cc_vals[(da, ta)] >= cc_vals[(db, tb)] else ((db, tb), (da, ta))
    add("crosscompare", f"Is the average current salary of employees with the title "
        f"{lo_side[1]} in the {lo_side[0]} department higher than that of employees with "
        f"the title {hi_side[1]} in the {hi_side[0]} department? Answer true or false.",
        "false", {"dept_hi": dm[hi_side[0]], "dept_lo": dm[lo_side[0]]})

    # ── finalize ─────────────────────────────────────────────────────────
    expected = {"lookup": 4, "multihop": 4, "aggregate": 4, "temporal": 4,
                "superlative": 2, "boundary": 16, "exclusion": 11, "disambiguation": 9,
                "comparison": 8, "arithmetic": 6, "verification": 7, "deep": 6,
                "rank": 6, "fanout": 3, "nested": 4, "history": 3, "crosscompare": 2}
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    assert counts == expected, f"category counts mismatch: {counts}"
    assert len(tasks) == 99, f"expected 99 tasks, got {len(tasks)}"
    assert all(t["gold_answer"] not in ("", "None") for t in tasks), "empty gold found"

    doc = {
        "dataset": "employees",
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": f"live API at {BASE} (pristine datacharmer/test_db v1.0.7)",
        "answer_format": (
            "numbers plain (no thousands separators, averages/percentages/ratios to 2 "
            "decimals); names 'First Last'; dates YYYY-MM-DD; booleans 'true'/'false'"
        ),
        "categories": {
            "lookup": "single-hop attribute of a named employee",
            "multihop": "2-hop entity chains across employees/departments/managers",
            "aggregate": "server-side computed statistics and filtered counts",
            "temporal": "point-in-time queries over history tables",
            "superlative": "server-side superlatives (top earners)",
            "boundary": "strict-inequality and period-inclusive counting traps over "
                        "populated boundaries — the naive inclusive or year-overflow "
                        "filter provably differs",
            "exclusion": "filtered count minus filtered count, incl. senior-title and "
                         "decade complements",
            "disambiguation": "attribute of the uniquely-qualified one among same-name "
                              "employees (stated, entity-referenced, or double "
                              "disambiguation)",
            "comparison": "true/false comparison of two retrieved values (near-miss, "
                          "strict-date boundaries)",
            "arithmetic": "differences/percentages/ratios computed over retrievals",
            "verification": "yes/no state claims over history and tenure (near-miss)",
            "deep": "3+ tool-call chains (dept -> top earner/manager -> attribute)",
            "rank": "n-th ranked positions, ranked exclusions, and top-n set logic",
            "fanout": "argmax over a whole-collection artifact (group_by/headcount)",
            "nested": "derived thresholds and dates chained into subsequent filters",
            "history": "penultimate/record-indexed lookups and growth booleans",
            "crosscompare": "near-miss comparisons between two computed populations",
        },
        "tasks": tasks,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote {len(tasks)} tasks -> {os.path.normpath(OUT_PATH)}\n")
    for t in tasks:
        q = t["question"] if len(t["question"]) <= 96 else t["question"][:93] + "..."
        print(f"{t['id']} [{t['category']:13s}] {q}")
        print(f"   -> {t['gold_answer']}")


if __name__ == "__main__":
    generate()
