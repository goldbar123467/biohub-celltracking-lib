"""Bounded Kaggle GPU/data/checkpoint smoke test; no competition training."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

# Verify and install the attached CPU-built wheel bundle without network access.
manifests = list(Path('/kaggle/input').glob('*/wheelhouse-manifest.json'))
manifests += list(Path('/kaggle/input').glob('notebooks/*/*/wheelhouse-manifest.json'))
assert len(manifests) == 1, 'Expected exactly one attached dependency bundle'
manifest_path = manifests[0]
manifest = json.loads(manifest_path.read_text())
assert manifest['python'] == list(sys.version_info[:2]), 'Wheel Python version mismatch'
wheel_dir = manifest_path.parent / 'wheels'
for file in manifest['files']:
    wheel = wheel_dir / file['name']
    assert wheel.stat().st_size == file['bytes']
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == file['sha256']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
                str(wheel_dir), *manifest['requirements']], check=True, timeout=90)

import numpy as np
import torch
import zarr


def deadline(_signal, _frame):
    raise TimeoutError('Readiness check exceeded 180 seconds')


signal.signal(signal.SIGALRM, deadline)
signal.alarm(180)
started = time.monotonic()
output = Path('/kaggle/working')
slug = 'biohub-cell-tracking-during-development'
root = next(p for p in (Path('/kaggle/input/competitions') / slug,
                       Path('/kaggle/input') / slug) if p.is_dir())
samples = sorted((root / 'train').glob('*.zarr'))
assert samples, 'No training samples mounted'
assert torch.cuda.is_available(), 'CUDA unavailable'
report = {'kind': 'infrastructure_smoke_only', 'data_root': str(root),
          'dependency_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'train_samples': len(samples),
          'test_samples': len(list((root / 'test').glob('*.zarr'))),
          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'zarr': importlib.metadata.version('zarr'), 'gpus': []}
torch.manual_seed(20260905)
for index in range(torch.cuda.device_count()):
    device = torch.device(f'cuda:{index}')
    layer = torch.nn.Conv3d(1, 2, 3, padding=1).to(device)
    x = torch.randn(1, 1, 8, 32, 32, device=device, requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    props = torch.cuda.get_device_properties(index)
    report['gpus'].append({'index': index, 'name': props.name,
                           'memory_bytes': props.total_memory,
                           'compute_capability': list(torch.cuda.get_device_capability(index)),
                           'forward_backward': 'passed'})

image = zarr.open_array(str(samples[0] / '0'), mode='r')
frame = np.asarray(image[0])
assert frame.dtype == np.uint16 and frame.ndim == 3
report.update(sample=samples[0].name, image_shape=list(image.shape),
              dtype=str(frame.dtype), frame_range=[int(frame.min()), int(frame.max())])
# Exercise serialization and reload using tensor-only data produced in this run.
checkpoint = output / 'smoke-checkpoint.pt'
state = {key: value.detach().cpu() for key, value in layer.state_dict().items()}
torch.save(state, checkpoint)
reloaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert state.keys() == reloaded.keys()
assert all(torch.equal(state[key], reloaded[key]) for key in state)
report.update(checkpoint_roundtrip='passed',
              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              elapsed_seconds=round(time.monotonic() - started, 3), status='passed')
(output / 'cloud-readiness.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
signal.alarm(0)
