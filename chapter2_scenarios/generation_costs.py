"""Duration and token use of the generation sessions, read from the Codex session records.

The generation ran as named sub-agent threads of one Codex desktop thread (records in ~/.codex/sessions/2026/08/18 and
/21). For each session the duration is the span of its record timestamps and the tokens are the final cumulative
token_count of its own thread (input including cached input, output including reasoning). The draft review started by
the round-2 roots session is listed separately and counted with the asymmetric procedure.
Writes results/generation_costs.csv.
"""
import csv
import glob
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = Path.home() / '.codex' / 'sessions' / '2026' / '08'
PLAN = [('initial_roles_a', 'initial'), ('initial_roles_b', 'initial'), ('initial_roles_c', 'initial'),
        ('targeted_round1_a', 'round 1'), ('targeted_round1_b', 'round 1'), ('targeted_round1_c', 'round 1'),
        ('targeted_round2_bridge', 'round 2'), ('targeted_round2_roots', 'round 2'),
        ('targeted_round2_roots/continuity_draft_review', 'round 2 (draft review)'),
        ('targeted_round2_identity', 'round 2'), ('targeted_round3_control_service', 'round 3'),
        ('admin_entry_refinement', 'round 4'), ('direct_baseline_ultra', 'direct')]


def main():
    found = {}
    for f in sorted(glob.glob(str(SESSIONS / '18' / '*.jsonl')) + glob.glob(str(SESSIONS / '21' / '*.jsonl'))):
        recs = []
        for line in open(f, encoding='utf-8'):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        path = (recs[0].get('payload', {}).get('agent_path') or '').replace('/root/', '')
        if path not in dict(PLAN):
            continue
        ts = [datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) for r in recs if 'timestamp' in r]
        usage = [r['payload']['info']['total_token_usage'] for r in recs if r.get('type') == 'event_msg'
                 and isinstance(r.get('payload'), dict) and r['payload'].get('type') == 'token_count'
                 and r['payload'].get('info')][-1]
        ctx = [r['payload'] for r in recs if r.get('type') == 'turn_context'][0]
        found[path] = {'session': path, 'stage': dict(PLAN)[path], 'model': ctx.get('model'), 'effort': ctx.get('effort'),
                       'start_utc': min(ts).isoformat(timespec='seconds'),
                       'minutes': round((max(ts) - min(ts)).total_seconds() / 60, 1),
                       'input_tokens': usage['input_tokens'], 'cached_input_tokens': usage['cached_input_tokens'],
                       'output_tokens': usage['output_tokens'], 'reasoning_output_tokens': usage['reasoning_output_tokens'],
                       'total_tokens': usage['total_tokens'], 'record_file': Path(f).name}
    rows = [found[p] for p, _ in PLAN]
    with open(ROOT / 'results' / 'generation_costs.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    asym = [r for r in rows if r['stage'] != 'direct']
    main_sessions = [r for r in asym if 'draft' not in r['stage']]
    print(f"asymmetric: {len(main_sessions)} sessions + draft review; {min(r['minutes'] for r in main_sessions)}-"
          f"{max(r['minutes'] for r in main_sessions)} min each; {sum(r['minutes'] for r in asym):.1f} min in total; "
          f"{sum(r['total_tokens'] for r in asym):,} tokens ({sum(r['output_tokens'] for r in asym):,} output)")
    d = [r for r in rows if r['stage'] == 'direct'][0]
    print(f"direct: {d['minutes']} min; {d['total_tokens']:,} tokens ({d['output_tokens']:,} output)")


if __name__ == '__main__':
    main()
