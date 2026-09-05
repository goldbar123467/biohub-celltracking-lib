"""Check CUDA execution and decode one completed training sample, without training."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
import zarr


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.manual_seed(20260905)
    layer = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 1, 8, 16, 16, device='cuda', requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
        raise RuntimeError('Nonfinite CUDA smoke result')
    report = {
        'cuda_smoke': 'passed', 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'kaggle': importlib.metadata.version('kaggle'),
        'kagglehub': importlib.metadata.version('kagglehub'),
        'zarr': zarr.__version__, 'loss': float(loss.detach().cpu()),
        'data_smoke': 'pending_complete_training_sample',
    }
    for metadata in sorted(Path('data/train').glob('*.zarr/0/zarr.json')):
        image_path = metadata.parent.parent
        graph_path = image_path.with_suffix('.geff')
        if not (graph_path / 'zarr.json').exists():
            continue
        image = zarr.open_array(str(metadata.parent), mode='r')
        # Metadata comes after chunks in the archive; this frame should be complete.
        if not (metadata.parent / 'c/0/0/0/0').is_file():
            continue
        frame = np.asarray(image[0])
        graph = zarr.open_group(str(graph_path), mode='r')
        ids = np.asarray(graph['nodes/ids'][:])
        edges = np.asarray(graph['edges/ids'][:])
        if frame.ndim != 3 or frame.dtype != np.uint16 or not np.isfinite(frame).all():
            raise ValueError('Unexpected microscopy frame contract')
        if ids.ndim != 1 or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError('Unexpected GEFF graph contract')
        if len(np.unique(ids)) != len(ids) or not np.isin(edges, ids).all():
            raise ValueError('Invalid graph IDs or edge references')
        report.update(data_smoke='passed', sample=image_path.name,
                      axes='T,Z,Y,X', image_shape=list(image.shape),
                      frame_dtype=str(frame.dtype), frame_range=[int(frame.min()), int(frame.max())],
                      annotated_nodes=len(ids), annotated_edges=len(edges))
        break
    Path('reports/environment-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
