"""AHP ranking with the revised ratings (29 September 2026).

The frozen reference model (scenario/reference_model.json) keeps the original judgment set, because its hash binds
the node mappings. Three ratings that contradicted the rating anchors were revised (scenario/ahp_ratings_revised.json):
remote service feasibility 6 -> 7 (exposed entry), maintenance provider feasibility 4 -> 7 (trusted entry) and
enterprise endpoint likelihood 6 -> 7 (documented entry). This script recomputes the scores, the weight sensitivity
with the procedure and seed of run_experiment.py (10,000 runs, multipliers U(0.8, 1.2), renormalized, seed
20260818), the one-point rating check, and the supplier-evidence scenario (maintenance provider likelihood 7).
Writes results/ahp_revised.json.
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ['plausibility', 'causal_directness', 'low_adversary_effort', 'low_early_detectability']


def weights(matrix):
    vals, vecs = np.linalg.eig(np.asarray(matrix, dtype=float))
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals.real[k])


def scores(w, ratings):
    return {f: float(sum(w[i] * r[c] / 9 for i, c in enumerate(CRITERIA))) for f, r in ratings.items()}


def sensitivity(w, ratings, repetitions=10000, seed=20260818):
    rng = np.random.default_rng(seed)
    fams = list(ratings)
    base_order = sorted(fams, key=lambda f: -scores(w, ratings)[f])
    top = {f: 0 for f in fams}
    same_rank = {f: 0 for f in fams}
    for _ in range(repetitions):
        ww = w * rng.uniform(0.8, 1.2, len(w))
        ww /= ww.sum()
        s = scores(ww, ratings)
        order = sorted(fams, key=lambda f: -s[f])
        top[order[0]] += 1
        for pos, f in enumerate(base_order):
            same_rank[f] += order[pos] == f
    return {'repetitions': repetitions, 'seed': seed, 'top_share': {f: n / repetitions for f, n in top.items()},
            'same_rank_share': {f: n / repetitions for f, n in same_rank.items()}}


def main():
    ref = json.loads((ROOT / 'scenario' / 'reference_model.json').read_text(encoding='utf-8'))
    ahp = ref['ahp']
    rev = json.loads((ROOT / 'scenario' / 'ahp_ratings_revised.json').read_text(encoding='utf-8'))
    w, lam = weights(ahp['pairwise_matrix'])
    ratings = {f: dict(r) for f, r in rev['family_ratings'].items()}
    base = scores(w, ratings)
    out = {'weights': dict(zip(CRITERIA, w.round(4).tolist())), 'lambda_max': round(lam, 4),
           'scores': {f: round(v, 3) for f, v in sorted(base.items(), key=lambda x: -x[1])},
           'sensitivity': sensitivity(w, ratings)}
    leader = max(base, key=base.get)
    flips = []
    for f, c, d in itertools.product(ratings, CRITERIA, (-1, 1)):
        if c == 'causal_directness':
            continue
        r = {k: dict(v) for k, v in ratings.items()}
        r[f][c] = min(9, max(1, r[f][c] + d))
        s = scores(w, r)
        if max(s, key=s.get) != leader:
            flips.append((f, c, d))
    out['one_point_changes_altering_first_place'] = flips
    r = {k: dict(v) for k, v in ratings.items()}
    r[leader]['plausibility'] = 7
    out['leader_with_likelihood_7'] = round(scores(w, r)[leader], 3)
    r = {k: dict(v) for k, v in ratings.items()}
    r['provider_identity']['plausibility'] = 7
    s = scores(w, r)
    out['supplier_scenario'] = {'provider_score': round(s['provider_identity'], 3),
                                'order': sorted(s, key=lambda f: -s[f]),
                                'sensitivity': sensitivity(w, r)}
    (ROOT / 'results' / 'ahp_revised.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
