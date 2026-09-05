"""Independently compare downloaded files with the frozen Kaggle API inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha256', action='store_true', help='Also rehash files against ZIP extraction receipts')
    args = parser.parse_args()
    root = Path('data')
    expected = json.loads(Path('reports/competition-files.json').read_text())
    status = json.loads(Path('reports/download-status.json').read_text())
    errors = []
    if status['state'] != 'complete':
        raise SystemExit('Dataset download is not complete; full verification deferred')
    receipts = {}
    if args.sha256:
        with Path('reports/download-files.jsonl').open() as stream:
            receipts = {row['path']: row for row in map(json.loads, stream)}
    for item in expected:
        path = root / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            errors.append(item['path'])
            continue
        if args.sha256:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != receipts.get(item['path'], {}).get('sha256'):
                errors.append(item['path'])
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    extras = sorted(actual - {item['path'] for item in expected})
    result = {'verified': not errors and not extras, 'files_expected': len(expected),
              'files_present': len(actual), 'sha256_checked': args.sha256,
              'mismatch_count': len(errors), 'extra_count': len(extras),
              'mismatch_examples': errors[:10], 'extra_examples': extras[:10]}
    Path('reports/data-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
