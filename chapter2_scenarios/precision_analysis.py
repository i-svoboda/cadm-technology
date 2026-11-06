"""Precision of generation: share of generated elements with an equivalent reference element.

The recorded mapping (scenario/node_mapping_overrides.json for the asymmetric outputs, scenario/direct_joint_ultra_mapping.json
for the direct baseline) assigns each generated node to an equivalent reference node or to null. The given endpoints
(A, the initial state, and B, the target incident) are excluded. Stages are cumulative, as in run_experiment.py:
initial trees, then the trees of refinement rounds 1-4. A node is identified by its id and label; revised trees that
repeat a node are counted once. Unmatched nodes are classified by position:
  objects: "part of a matched object" if an ancestor other than A is matched, otherwise "separate branch";
  events: "next to a matched event" if a direct cause or effect other than B is matched, otherwise "separate branch".
Writes results/precision_by_stage.csv and results/unmatched_elements.csv.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs_v2'
STAGES = [('initial', ['initial']), ('round 1', ['targeted_round1']), ('round 2', ['targeted_round2']),
          ('round 3', ['targeted_round3']), ('round 4', ['targeted_round4'])]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def namespace(folder, name):
    return name if folder == 'initial' else f'{folder}/{name}'


def classify(tree, omap, emap):
    """Unmatched nodes of one tree with their positional class."""
    parents = {e['child']: e['parent'] for e in tree['object_edges']}
    causes, effects = {}, {}
    for e in tree['causal_edges']:
        effects.setdefault(e['cause'], []).append(e['effect'])
        causes.setdefault(e['effect'], []).append(e['cause'])
    out = {}
    for n in tree['object_nodes']:
        if n['node_type'] == 'state_root' or omap.get(n['id']):
            continue
        p, cls, seen = parents.get(n['id']), 'separate branch', set()
        while p and p not in seen and p != 'A':
            seen.add(p)
            if omap.get(p):
                cls = 'part of a matched object'
                break
            p = parents.get(p)
        out[('object', n['id'], n['label'])] = (cls, n.get('node_type'))
    for n in tree['event_nodes']:
        if n['node_type'] == 'target_event' or emap.get(n['id']):
            continue
        nb = [x for x in causes.get(n['id'], []) + effects.get(n['id'], []) if x != 'B']
        cls = 'next to a matched event' if any(emap.get(x) for x in nb) else 'separate branch'
        out[('event', n['id'], n['label'])] = (cls, n.get('node_type'))
    return out


def tally(trees):
    """trees: list of (tree, omap, emap). Distinct nodes by (id, label), excluding A and B."""
    objs, evs, unmatched = {}, {}, {}
    for tree, omap, emap in trees:
        for n in tree['object_nodes']:
            if n['node_type'] != 'state_root':
                key = (n['id'], n['label'])
                objs[key] = objs.get(key, False) or bool(omap.get(n['id']))
        for n in tree['event_nodes']:
            if n['node_type'] != 'target_event':
                key = (n['id'], n['label'])
                evs[key] = evs.get(key, False) or bool(emap.get(n['id']))
        unmatched.update(classify(tree, omap, emap))
    unmatched = {k: v for k, v in unmatched.items()
                 if not (objs if k[0] == 'object' else evs).get((k[1], k[2]))}
    return objs, evs, unmatched


def main():
    mapping = load(ROOT / 'scenario' / 'node_mapping_overrides.json')
    om, em = mapping['object_node_mappings'], mapping['event_node_mappings']
    rows, cumulative, final_unmatched = [], [], {}
    direct = load(RUNS / 'direct_joint_ultra' / 'direct_joint_generation.json')
    dmap = load(ROOT / 'scenario' / 'direct_joint_ultra_mapping.json')
    d_om = dmap.get('object_node_mappings', dmap.get('objects', {}))
    d_em = dmap.get('event_node_mappings', dmap.get('events', {}))
    if 'direct_joint_ultra/direct_joint_generation.json' in d_om:
        d_om = d_om['direct_joint_ultra/direct_joint_generation.json']
        d_em = d_em['direct_joint_ultra/direct_joint_generation.json']
    stages = [('direct', [(direct, d_om, d_em)])]
    for label, folders in STAGES:
        for folder in folders:
            for f in sorted((RUNS / folder).glob('*.json')):
                ns = namespace(folder, f.name)
                cumulative.append((load(f), om[ns], em[ns]))
        stages.append((label, list(cumulative)))
    for label, trees in stages:
        objs, evs, unmatched = tally(trees)
        c = {cls: sum(1 for (kind, _, _), (k, _) in unmatched.items() if kind == kind_ and k == cls)
             for kind_, cls in [('object', 'part of a matched object'), ('object', 'separate branch'),
                                ('event', 'next to a matched event'), ('event', 'separate branch')]}
        rows.append({'stage': label, 'objects': len(objs), 'objects_matched': sum(objs.values()),
                     'events': len(evs), 'events_matched': sum(evs.values()),
                     'unmatched_objects_part_of_matched': sum(1 for (k, _, _), (cl, _) in unmatched.items()
                                                              if k == 'object' and cl == 'part of a matched object'),
                     'unmatched_objects_separate': sum(1 for (k, _, _), (cl, _) in unmatched.items()
                                                       if k == 'object' and cl == 'separate branch'),
                     'unmatched_events_next_to_matched': sum(1 for (k, _, _), (cl, _) in unmatched.items()
                                                             if k == 'event' and cl == 'next to a matched event'),
                     'unmatched_events_separate': sum(1 for (k, _, _), (cl, _) in unmatched.items()
                                                      if k == 'event' and cl == 'separate branch')})
        if label in ('direct', 'round 4'):
            for (kind, nid, lab), (cls, ntype) in unmatched.items():
                final_unmatched[(label, kind, nid, lab)] = (cls, ntype)
    with open(ROOT / 'results' / 'precision_by_stage.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / 'results' / 'unmatched_elements.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['condition', 'kind', 'id', 'label', 'position', 'node_type'])
        for (label, kind, nid, lab), (cls, ntype) in sorted(final_unmatched.items()):
            w.writerow([label, kind, nid, lab, cls, ntype])
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
