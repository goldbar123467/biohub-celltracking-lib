"""Download only the small training graph needed by upstream metric tests."""
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    prefix = 'train/6bba_c328f2fd.geff/'
    files = [f for f in json.loads(Path('reports/competition-files.json').read_text())
             if f['path'].startswith(prefix)]
    if not files or sum(f['bytes'] for f in files) > 10_000_000:
        raise ValueError('Missing or unexpectedly large metric fixture')
    api = KaggleApi()
    api.authenticate()
    for file in files:
        target = Path('data') / file['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        api.competition_download_file('biohub-cell-tracking-during-development',
                                      file['path'], path=str(target.parent), quiet=True)
        if target.stat().st_size != file['bytes']:
            raise ValueError('Fixture download size mismatch')
    print(json.dumps({'fixture_files': len(files),
                      'fixture_bytes': sum(f['bytes'] for f in files)}))


if __name__ == '__main__':
    main()
