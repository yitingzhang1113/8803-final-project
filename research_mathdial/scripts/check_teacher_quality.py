#!/usr/bin/env python3
import argparse
import json
from typing import Dict


def read_jsonl(p: str):
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def index_deps(path: str) -> Dict[tuple, dict]:
    idx = {}
    for d in read_jsonl(path):
        idx[(d['qid'], d['turn_id'])] = d
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deps', required=True, help='Path to dependencies jsonl from A')
    ap.add_argument('--teacher', required=True, help='Path to teacher signals jsonl from B')
    args = ap.parse_args()

    deps = index_deps(args.deps)

    total = 0
    miss_ref = 0
    bad_ref_future = 0
    match_dep = 0
    hint_dist: Dict[str, int] = {}
    conf_sum = 0.0
    sol_steps = 0
    sol_ans = 0

    for o in read_jsonl(args.teacher):
        total += 1
        qid = o.get('qid')
        tid = o.get('turn_id')
        ref = o.get('ref_turn')
        hint = o.get('hint_type', '')
        hint_dist[hint] = hint_dist.get(hint, 0) + 1
        try:
            conf_sum += float(o.get('confidence', 0.0))
        except Exception:
            pass
        if o.get('solution_steps'):
            sol_steps += 1
        if o.get('final_answer'):
            sol_ans += 1

        dep = deps.get((qid, tid))
        if dep:
            a_dep = dep.get('depends_on')
            # Future ref invalid
            if isinstance(ref, int) and isinstance(tid, int) and ref >= tid:
                bad_ref_future += 1
            if ref is None:
                miss_ref += 1
            if isinstance(a_dep, int) and a_dep == ref:
                match_dep += 1
        else:
            # Without A record, we can only check presence
            if ref is None:
                miss_ref += 1

    avg_conf = round(conf_sum / total, 3) if total else 0.0
    print(json.dumps({
        'total': total,
        'hint_dist': hint_dist,
        'avg_confidence': avg_conf,
        'ref_turn_missing': miss_ref,
        'ref_turn_future_invalid': bad_ref_future,
        'ref_turn_matches_dep': match_dep,
        'with_solution_steps': sol_steps,
        'with_final_answer': sol_ans,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
