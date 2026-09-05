"""CPU-only build of the small wheel bundle required to decode Biohub data offline."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/kaggle/working')
wheels = root / 'wheels'
wheels.mkdir(exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                '--dest', str(wheels), 'zarr==3.3.0', 'numcodecs==0.15.1'],
               check=True, timeout=180)
manifest = {'python': list(sys.version_info[:2]), 'requirements': ['zarr==3.3.0', 'numcodecs==0.15.1'],
            'files': [{'name': p.name, 'bytes': p.stat().st_size,
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(wheels.glob('*.whl'))]}
(root / 'wheelhouse-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
