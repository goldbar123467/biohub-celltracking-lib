"""Stream the authenticated competition ZIP into files without keeping the ZIP.

Atomic per-file writes, ZIP CRC validation, SHA-256 receipts, disk reserve, and
bounded retries. A retry replays the ZIP from the beginning; it is not HTTP resume.
No token, signed URL, response headers, or exception text is logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
import requests
from stream_unzip import stream_unzip

SLUG = 'biohub-cell-tracking-during-development'


def save_status(path: Path, values: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(values, indent=2) + '\n')
    temp.replace(path)


def target_path(root: Path, name: bytes) -> Path:
    relative = PurePosixPath(name.decode('utf-8'))
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
        raise ValueError('Unsafe archive path')
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Archive path escapes output directory')
    return target


def download(root: Path, reports: Path, reserve_gib: int, expected_files: int,
             max_files: int, attempt: int) -> None:
    api = KaggleApi()
    api.authenticate()
    started = time.monotonic()
    last_report = started
    count = total = network = 0
    seen: set[str] = set()
    status_path = reports / 'download-status.json'
    receipt_path = reports / ('smoke-files.jsonl' if max_files else 'download-files.jsonl')
    status = {'state': 'running', 'competition': SLUG, 'attempt': attempt,
              'files': 0, 'bytes_extracted': 0, 'expected_files': expected_files}
    save_status(status_path, status)

    with api.build_kaggle_client() as client:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = SLUG
        response = client.competitions.competition_api_client.download_data_files(request)
        with response, receipt_path.open('w') as receipt:
            response.raise_for_status()

            def chunks():
                nonlocal network
                for chunk in response.iter_content(1024 * 1024):
                    network += len(chunk)
                    yield chunk

            for raw_name, size, contents in stream_unzip(chunks()):
                target = target_path(root, raw_name)
                if raw_name.endswith(b'/'):
                    target.mkdir(parents=True, exist_ok=True)
                    for chunk in contents:
                        if chunk:
                            raise ValueError('Nonempty directory entry')
                    continue
                relative = target.relative_to(root).as_posix()
                if relative in seen:
                    raise ValueError('Duplicate archive member')
                seen.add(relative)
                if shutil.disk_usage(root).free < reserve_gib * 2**30 + size:
                    raise OSError('Disk reserve reached')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                digest = hashlib.sha256()
                written = 0
                with temporary.open('wb') as output:
                    for chunk in contents:
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                # Exhausting contents also verifies the ZIP member's CRC.
                if written != size:
                    raise ValueError('Archive member size mismatch')
                os.replace(temporary, target)
                receipt.write(json.dumps({'path': relative, 'bytes': written,
                                          'sha256': digest.hexdigest()}) + '\n')
                count += 1
                total += written
                now = time.monotonic()
                if now - last_report > 15 or count == 1:
                    status.update(files=count, bytes_extracted=total, network_bytes=network,
                                  elapsed_seconds=round(now-started, 1), last_file=relative)
                    save_status(status_path, status)
                    receipt.flush()
                    print(json.dumps(status), flush=True)
                    last_report = now
                if max_files and count >= max_files:
                    break

    if not max_files and count != expected_files:
        raise ValueError('Archive file count differs from frozen data-page count')
    status.update(state='smoke_complete' if max_files else 'complete', files=count,
                  bytes_extracted=total, network_bytes=network,
                  elapsed_seconds=round(time.monotonic()-started, 1))
    save_status(status_path, status)
    print(json.dumps(status), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--reports-dir', type=Path, default=Path('reports'))
    parser.add_argument('--reserve-gib', type=int, default=15)
    parser.add_argument('--expected-files', type=int, default=24886)
    parser.add_argument('--max-files', type=int, default=0)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            download(root, args.reports_dir, args.reserve_gib,
                     args.expected_files, args.max_files, attempt)
            return
        except Exception as error:
            # Network exceptions can embed signed URLs. Save only type and code.
            status = {'state': 'failed', 'attempt': attempt,
                      'error_type': type(error).__name__}
            if isinstance(error, requests.HTTPError) and error.response is not None:
                status['http_status'] = error.response.status_code
            save_status(args.reports_dir / 'download-status.json', status)
            print(json.dumps(status), flush=True)
            if attempt == 3 or not isinstance(error, requests.RequestException):
                raise SystemExit(1) from None
            time.sleep(5 * attempt)


if __name__ == '__main__':
    main()
