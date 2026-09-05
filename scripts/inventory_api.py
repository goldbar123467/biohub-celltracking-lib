"""Freeze the complete paginated competition file inventory without downloading data."""
from __future__ import annotations

import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    files = []
    token = None
    pages = 0
    while True:
        response = api.competition_list_files(
            'biohub-cell-tracking-during-development', page_token=token, page_size=200
        )
        files.extend({'path': f.name, 'bytes': f.total_bytes,
                      'created': f.creation_date.isoformat() if f.creation_date else None}
                     for f in response.files)
        pages += 1
        if pages % 20 == 0:
            print(f'Inventory pages={pages} files={len(files)}', flush=True)
        token = response.next_page_token
        if not token:
            break
    if len({f['path'] for f in files}) != len(files):
        raise ValueError('Duplicate file paths in API inventory')
    summary = {'file_count': len(files), 'total_bytes': sum(f['bytes'] for f in files),
               'pages': pages}
    reports = Path('reports')
    reports.mkdir(exist_ok=True)
    (reports / 'competition-files.json').write_text(json.dumps(files, indent=2) + '\n')
    (reports / 'competition-files-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
