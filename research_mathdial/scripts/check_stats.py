"""Check statistics of dependency extraction output"""
import json

with open('results/dependencies_train.jsonl', 'r', encoding='utf-8') as f:
    ops = {}
    total = 0
    root_count = 0
    
    for line in f:
        d = json.loads(line)
        op = d.get('operation', 'other')
        ops[op] = ops.get(op, 0) + 1
        total += 1
        if d.get('depends_on') is None:
            root_count += 1
    
    print(f'Total entries: {total}')
    print(f'Root steps (depends_on=None): {root_count} ({root_count/total*100:.1f}%)')
    print('\nOperation distribution:')
    for op, count in sorted(ops.items(), key=lambda x: -x[1]):
        print(f'  {op}: {count} ({count/total*100:.1f}%)')

