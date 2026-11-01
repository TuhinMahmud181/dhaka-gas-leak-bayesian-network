"""Analyses added in the first-round revision (PSEP-D-26-08566).

Run from code/, after build_model.py:

    python revision_analyses.py det                 # Table S15 (one anchor at a time), Table S16, Dhaka counts
    python revision_analyses.py thresh              # riser-link values at which the bundle ranking reverses
    python revision_analyses.py t6                  # Table 6
    python revision_analyses.py prev                # Table S15, last row (anchors at the experts' prevalence answers)
    python revision_analyses.py mc anchors 2000 20261021   # Table S15, anchors drawn independently
    python revision_analyses.py mc joint 2000 20261020     # anchors and link probabilities together (Sections 3.1, 3.2, 3.6)
    python revision_analyses.py experts             # Tables S13 and S14 (needs the private questionnaire file)

Each command prints its results and writes out_<command>.json in code/.

The anchoring step is solved with calibrate() below, an exact and faster version of
build_model.calibrate: each node's marginal is computed on the grid of its own root
ancestors only, which gives the same exponents.

The 'experts' command needs the individual questionnaire answers, which are not public.
Set QUESTIONNAIRE_XLSX to the private file; it also needs openpyxl and krippendorff.
"""
import sys, os, copy, json, itertools
import numpy as np
import build_model as B
import leak_comparison as L

MAINS = ['External_Corrosion_', 'Outside_Force_Damage', 'Unauthorized_Modification_',
         'Material_Degradation_', 'Operational_Failure']
ORIG_ELICITED = copy.deepcopy(B.ELICITED)
ORIG_PARAMS = copy.deepcopy(B.PARAMS)
ORIG_TARGETS = dict(B.TARGETS)
KM_DHAKA = 7000
RISERS_DHAKA = L.N_RISERS * KM_DHAKA / L.NET_KM_TITAS
COMPLAINTS = 4891          # annual leak complaints, Titas, FY 2021-22


# ---------------------------------------------------------------- anchoring (exact, fast)
def _roots(nodes, ch):
    out = set()
    for p in nodes[ch]['parents']:
        if nodes[p]['parents']: out |= set(_roots(nodes, p))
        else: out.add(p)
    return sorted(out)


_G = {}
def _grid(nodes, roots):
    key = tuple(roots)
    if key in _G: return _G[key]
    idx = [np.arange(len(nodes[r]['states'])) for r in roots]
    pr = [np.array(nodes[r]['probs']) for r in roots]
    g = np.meshgrid(*idx, indexing='ij'); wg = np.meshgrid(*pr, indexing='ij')
    A = {r: x.ravel() for r, x in zip(roots, g)}; w = np.ones(g[0].size)
    for x in wg: w = w * x.ravel()
    _G[key] = (A, w); return A, w


def _p(nodes, links, ch, A, fixed=None):
    fixed = fixed or {}
    n = len(next(iter(A.values())))
    q = np.full(n, 1.0 - B.PARAMS[ch]['leak'])
    for par, tab in links[ch].items():
        if par in fixed: q = q * (1.0 - tab['Yes' if fixed[par] == 0 else 'No'])
        else:
            lut = np.array([tab[s] for s in nodes[par]['states']]); q = q * (1.0 - lut[A[par]])
    return 1.0 - q


def node_marginal(nodes, links, ch):
    A, w = _grid(nodes, _roots(nodes, ch))
    if ch == 'Material_Degradation_':
        ec = _p(nodes, links, 'External_Corrosion_', A)
        m = ec * _p(nodes, links, ch, A, {'External_Corrosion_': 0}) + (1 - ec) * _p(nodes, links, ch, A, {'External_Corrosion_': 1})
        return float(np.dot(w, m))
    return float(np.dot(w, _p(nodes, links, ch, A)))


def calibrate(nodes, links, targets=None):
    targets = B.TARGETS if targets is None else targets
    alphas = {}
    for ch in B.PARAMS:
        if targets.get(ch) is None: continue
        base = {p: dict(t) for p, t in links[ch].items()}
        lo, hi = 1e-3, 60.0
        for _ in range(50):
            mid = (lo + hi) / 2
            links[ch] = {p: {s: 1 - (1 - v) ** mid for s, v in t.items()} for p, t in base.items()}
            if node_marginal(nodes, links, ch) < targets[ch]: lo = mid
            else: hi = mid
        a = (lo + hi) / 2; alphas[ch] = a
        links[ch] = {p: {s: 1 - (1 - v) ** a for s, v in t.items()} for p, t in base.items()}
    return alphas


# ---------------------------------------------------------------- helpers
def reset():
    B.PARAMS = copy.deepcopy(ORIG_PARAMS); B.ELICITED.clear(); B.ELICITED.update(copy.deepcopy(ORIG_ELICITED))
    B.TARGETS.clear(); B.TARGETS.update(ORIG_TARGETS); B._GRID_CACHE.clear()


def model(targets=None, elicited=None, params=None):
    """Load the network with optional changes, anchor it and return (nodes, links)."""
    reset()
    if params is not None: B.PARAMS = params
    if elicited is not None: B.ELICITED.clear(); B.ELICITED.update(copy.deepcopy(elicited))
    tree, nd = B.load(B.SRC)
    links = {c: {p: dict(t) for p, t in s['links'].items()} for c, s in B.PARAMS.items()}
    B.apply_elicited(links)
    calibrate(nd, links, B.TARGETS if targets is None else targets)
    return nd, links


def summary(nd, links):
    b = B.marginals(nd, links)
    mains, ris, tot = L.counts(b)
    out = {'PF': b['Pipeline_Failure'], 'RF': b['Riser_Failure'], 'dens': L.density(b['Pipeline_Failure']),
           'mains': mains, 'risers': ris, 'total': tot, 'ratio': tot / COMPLAINTS, 'avoided': {}}
    for i, lab, ev in B.scenarios(nd):
        if i != 1: out['avoided'][i] = tot - L.counts(B.marginals(nd, links, ev))[2]
    return out


def run(**kw):
    s = summary(*model(**kw)); reset(); return s


def riser_params(fic, fage):
    """Authored riser links of Illegal Connection and Pipeline Age multiplied by fic and fage."""
    p = copy.deepcopy(ORIG_PARAMS); lk = p['Riser_Failure']['links']
    for s in lk['Illegal_Connection_']: lk['Illegal_Connection_'][s] *= fic
    for s in lk['Pipeline_Age_']: lk['Pipeline_Age_'][s] *= fage
    return p


def fmt(s):
    a = s['avoided']
    return (f"PF {100 * s['PF']:5.1f}%  {s['dens']:.2f}/km  mains {s['mains']:>7,.0f} total {s['total']:>8,.0f} "
            f"ratio {s['ratio']:5.1f}  tech {a[7]:>7,.0f} man {a[8]:>7,.0f} age-repl {a[6]:>7,.0f}")


def save(name, obj):
    json.dump(obj, open(f'out_{name}.json', 'w'), indent=1, default=float)


# ---------------------------------------------------------------- analyses
def det():
    """Anchors one at a time, age links, riser links, leak term, Dhaka-only counts."""
    R = {}
    R['base'] = base = run(); print('base           ', fmt(base))
    print('\n1. one anchor at a time, x0.5 and x1.5'); R['oat'] = {}
    for h in MAINS:
        for f in (0.5, 1.5):
            t = dict(ORIG_TARGETS); t[h] = min(0.95, t[h] * f)
            R['oat'][f'{h}|{f}'] = s = run(targets=t); print(f'  {B.LABEL[h]:<26} x{f}  ', fmt(s))
    print('\n2. age links (Q13, Q19 asked about 40+ years; Old = past 30 years)'); R['age'] = {}
    for f in (0.5, 0.75, 1.0):
        e = copy.deepcopy(ORIG_ELICITED)
        for k in ('External_Corrosion_|Pipeline_Age_|Old', 'Material_Degradation_|Pipeline_Age_|Old'): e[k]['median'] *= f
        R['age'][f] = s = run(elicited=e); print(f'  age links x{f:<5}', fmt(s))
    print('\n3. riser links'); R['riser'] = {}
    for fic, fage in [(1, 1), (0.75, 1), (0.6, 1), (0.55, 1), (0.5, 1), (0.4, 1), (0, 1),
                      (1, 1.5), (1, 1.75), (1, 2), (1, 2.5), (1, 0)]:
        R['riser'][f'{fic}|{fage}'] = s = run(params=riser_params(fic, fage))
        print(f'  IC x{fic:<5} age x{fage:<5}', fmt(s), '->', 'managerial' if s['avoided'][8] > s['avoided'][7] else 'technical')
    print('\n4. leak term of Pipeline Leak'); R['lam'] = {}
    for lam in (0.005, 0.02, 0.05, 0.10):
        p = copy.deepcopy(ORIG_PARAMS); p['Pipeline_Failure']['leak'] = lam
        R['lam'][lam] = s = run(params=p); print(f'  leak term {lam:<6}', fmt(s))
    print('\n5. Dhaka only (7,000 km; risers in proportion to length)')
    d = {'mains': base['dens'] * KM_DHAKA, 'risers': base['RF'] * RISERS_DHAKA}; d['total'] = d['mains'] + d['risers']; R['dhaka'] = d
    print(f"  risers {RISERS_DHAKA:,.0f}  new leaks: mains {d['mains']:,.0f}  risers {d['risers']:,.0f}  total {d['total']:,.0f}")
    save('det', R)


def thresh():
    """Riser-link multipliers at which the managerial and technical bundles avoid equal numbers of leaks."""
    def diff(fic, fage):
        a = run(params=riser_params(fic, fage))['avoided']; return a[8] - a[7]
    def bis(f, lo, hi):
        flo = f(lo)
        for _ in range(25):
            m = (lo + hi) / 2; fm = f(m)
            if (fm > 0) == (flo > 0): lo, flo = m, fm
            else: hi = m
        return (lo + hi) / 2
    ic = bis(lambda x: diff(x, 1), 0.5, 0.6); ag = bis(lambda x: diff(1, x), 1.75, 2.0)
    print('Illegal-connection threshold', round(ic, 3), ' Pipe-age threshold', round(ag, 3))
    save('thresh', {'ic': ic, 'age': ag})


def t6():
    """Table 6: the two bundles and the two leading single measures when the riser links change."""
    out = {}
    for fic, fage in [(1, 1), (0.75, 1), (0.6, 1), (0.5, 1), (0, 1), (1, 1.5), (1, 2), (1, 0)]:
        a = run(params=riser_params(fic, fage))['avoided']
        out[f'{fic:g}|{fage:g}'] = {'ic': a[3], 'maint': a[5], 'tech': a[7], 'man': a[8]}
        print(f'IC x{fic:<5} age x{fage:<4} IC {a[3]:>7,.0f} maint {a[5]:>7,.0f} tech {a[7]:>7,.0f} man {a[8]:>7,.0f}')
    save('t6', out)


def prev():
    """Anchors of External Corrosion and Unauthorized Modification at the experts' prevalence answers (Q29, Q30)."""
    out = {}
    for a in (0.10, 0.175, 0.25):
        t = dict(ORIG_TARGETS); t['External_Corrosion_'] = a; t['Unauthorized_Modification_'] = a
        out[str(a)] = s = run(targets=t)
        print(f"anchors {a}: PF {100 * s['PF']:.1f}% {s['dens']:.2f}/km/yr ratio {s['ratio']:.1f} "
              f"tech {s['avoided'][7]:,.0f} man {s['avoided'][8]:,.0f}")
    save('prev', out)


def mc(mode, n, seed):
    """Monte Carlo with the five mains anchors drawn uniformly on 0.5-1.5 x base, anchoring solved
    again for each draw; mode 'joint' also perturbs the link probabilities as in build_model.sweep."""
    rng = np.random.default_rng(seed)
    reset(); tree, nd = B.load(B.SRC)
    base = {c: {p: dict(t) for p, t in s['links'].items()} for c, s in B.PARAMS.items()}; B.apply_elicited(base)
    out = []
    for _ in range(n):
        t = dict(ORIG_TARGETS)
        for h in MAINS: t[h] = min(0.95, ORIG_TARGETS[h] * rng.uniform(0.5, 1.5))
        links = {c: {p: dict(x) for p, x in v.items()} for c, v in base.items()}
        calibrate(nd, links, t)
        if mode == 'joint': links = B.perturb(links, rng)
        m = B.marginals(nd, links); mm, rr, tt = L.counts(m)
        row = {'PF': m['Pipeline_Failure'], 'dens': L.density(m['Pipeline_Failure']), 'total': tt, 'mains': mm, 'RF': m['Riser_Failure']}
        if mode == 'joint':
            for i, ev in ((7, B.SCENARIOS[6][1]), (8, B.SCENARIOS[7][1])):
                row[f'av{i}'] = tt - L.counts(B.marginals(nd, links, ev))[2]
        out.append(row)
    save(f'mc_{mode}', out)
    v = lambda k: np.array([r[k] for r in out])
    for k in ('PF', 'dens', 'mains', 'total'): print(mode, k, np.percentile(v(k), [5, 50, 95]).round(3))
    print(mode, 'per complaint', np.percentile(v('total') / COMPLAINTS, [5, 50, 95]).round(2))
    if mode == 'joint': print('managerial > technical in', (v('av8') > v('av7')).mean())


def experts():
    """Leave-one-expert-out, alternative pooling rules and agreement (Tables S13, S14)."""
    import openpyxl, krippendorff
    SCALE = {'Very low': 0.05, 'Low': 0.15, 'Moderate': 0.35, 'High': 0.60, 'Very high': 0.85}
    QKEY = {5: 'Pipeline_Failure|External_Corrosion_|Yes', 6: 'Pipeline_Failure|Outside_Force_Damage|Yes',
            7: 'Pipeline_Failure|Unauthorized_Modification_|Yes', 8: 'Pipeline_Failure|Material_Degradation_|Yes',
            9: 'Pipeline_Failure|Operational_Failure|Yes', 11: 'External_Corrosion_|Cathodic_Protection__|Absent',
            12: 'External_Corrosion_|Soil_Corrosivity_|High', 13: 'External_Corrosion_|Pipeline_Age_|Old',
            14: 'External_Corrosion_|Water_logging_|Frequent', 15: 'Outside_Force_Damage|Third_Party_Interference_|High',
            16: 'Outside_Force_Damage|Burial_Depth_|Shallow', 17: 'Unauthorized_Modification_|Illegal_Connection_|High',
            18: 'Unauthorized_Modification_|Burial_Depth_|Shallow', 19: 'Material_Degradation_|Pipeline_Age_|Old',
            20: 'Material_Degradation_|Material_Quality_|Low', 21: 'Riser_Failure|Riser_Installation_Quality_|Poor',
            22: 'Riser_Failure|Maintenance_Frequency_|Rare'}
    ws = openpyxl.load_workbook(os.environ.get('QUESTIONNAIRE_XLSX', 'questionnaire_responses.xlsx'), data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True)); hdr = rows[0]
    col = {q: [i for i, h in enumerate(hdr) if str(h).startswith(f'Q{q}.')][0] for q in QKEY}
    exp = [{'role': r[2], 'years': r[3], 'ans': {q: SCALE.get(str(r[col[q]]).strip()) for q in QKEY}} for r in rows[1:]]
    vals = {q: [e['ans'][q] for e in exp] for q in QKEY}

    def pooled(v_by_q, agg='median', weights=None):
        out = {}
        for q, key in QKEY.items():
            v = [(x, w) for x, w in zip(v_by_q[q], weights or [1] * len(v_by_q[q])) if x is not None]
            xs = np.array([a for a, _ in v]); ws_ = np.array([b for _, b in v], float)
            if agg == 'median': m = float(np.median(xs))
            elif agg == 'mean': m = float(xs.mean())
            elif agg == 'geomean': m = float(np.exp(np.log(xs).mean()))
            elif agg == 'trimmed': s = np.sort(xs); m = float(s[1:-1].mean()) if len(s) > 2 else float(s.mean())
            elif agg == 'wmean': m = float((xs * ws_).sum() / ws_.sum())
            out[key] = {'median': m, 'lo': float(xs.min()), 'hi': float(xs.max()), 'n': len(xs)}
        return out

    el = pooled(vals)
    for k, v in el.items():   # the medians must reproduce Table 2
        o = ORIG_ELICITED[k]; assert abs(v['median'] - o['median']) < 1e-9 and v['n'] == o['n'], (k, v, o)
    print('medians reproduce Table 2: OK')
    res = {}
    def go(tag, elic):
        res[tag] = s = run(elicited=elic)
        print(f"{tag:<30} PF {100 * s['PF']:5.1f}%  {s['dens']:.2f}/km/yr  total {s['total']:>8,.0f}  ratio {s['ratio']:5.1f}  "
              f"tech {s['avoided'][7]:>7,.0f} man {s['avoided'][8]:>7,.0f}")
    go('base (median, n=7)', el)
    for i, e in enumerate(exp):
        go(f'without expert {i + 1} ({e["years"]})', pooled({q: [x for j, x in enumerate(vals[q]) if j != i] for q in QKEY}))
    for agg in ('mean', 'geomean', 'trimmed'): go(f'aggregation: {agg}', pooled(vals, agg))
    yw = {'Under 5': 1, '5-10': 2, '11-20': 3}
    go('experience-weighted mean', pooled(vals, 'wmean', [yw[e['years']] for e in exp]))
    senior = [i for i, e in enumerate(exp) if e['years'] != 'Under 5']
    go('only experts >=5 years (n=2)', pooled({q: [vals[q][i] for i in senior] for q in QKEY}))
    go('authors only', {})
    # agreement; "unsure" answers are left out
    lev = {0.05: 1, 0.15: 2, 0.35: 3, 0.60: 4, 0.85: 5}
    M = np.array([[lev[e['ans'][q]] if e['ans'][q] is not None else np.nan for q in QKEY] for e in exp])
    alpha = krippendorff.alpha(reliability_data=M, level_of_measurement='ordinal')
    def tw(x):   # Tastle-Wierman consensus
        x = x[~np.isnan(x)]; p = np.array([(x == k).mean() for k in range(1, 6)]); mu = (p * np.arange(1, 6)).sum()
        return 1 + sum(pk * np.log2(1 - abs(k - mu) / 4) for k, pk in zip(range(1, 6), p) if pk > 0)
    cons = {q: tw(M[:, j]) for j, q in enumerate(QKEY)}
    within1 = {}
    for j, q in enumerate(QKEY):
        x = M[:, j]; x = x[~np.isnan(x)]; within1[q] = float(np.mean(np.abs(x - np.median(x)) <= 1))
    print(f'Krippendorff ordinal alpha = {alpha:.3f}')
    for q in QKEY: print(f'  Q{q:<3} consensus {cons[q]:.2f}  within one step of median {100 * within1[q]:.0f}%')
    res['agreement'] = {'alpha': alpha, 'cons': cons, 'within1': within1}
    save('experts', res)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'mc': mc(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd in ('det', 'thresh', 't6', 'prev', 'experts'): globals()[cmd]()
    else: print(__doc__)
