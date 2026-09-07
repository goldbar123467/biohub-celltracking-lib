"""Export a frozen detector and build its private, offline Kaggle notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from package_kaggle_notebook import build


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_model(checkpoint, snapshot_path, output_dir, dataset_id):
    import torch

    from biohub_ct.pipelines.learned import LearnedConfig
    from biohub_ct.training.model import ModelConfig, PointDetector3D

    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8-sig"))
    selection = snapshot["reports"]["fold0/selection.json"]
    campaign = snapshot["reports"]["manifest.json"]
    if sha256(checkpoint) != selection["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint does not match the frozen selection")
    # This is our own training artifact, authenticated by the saved selection hash.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["step"] != selection["checkpoint"]["step"]:
        raise ValueError("Selected checkpoint step mismatch")
    if state["model_config"] != campaign["model"]:
        raise ValueError("Selected model configuration mismatch")
    model = PointDetector3D(ModelConfig(**state["model_config"])).eval()
    model.load_state_dict(state["model"], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("Nonfinite model weights")
    config = {**campaign["inference"], "threshold": selection["selected_threshold"]}
    LearnedConfig(**config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "detector-weights.pt"
    torch.save(model.state_dict(), weights)
    restored = torch.load(weights, map_location="cpu", weights_only=True)
    if not all(torch.equal(t, restored[k]) for k, t in model.state_dict().items()):
        raise RuntimeError("Inference export did not preserve every model tensor")
    manifest = {
        "format_version": 1,
        "weights_file": weights.name,
        "weights_sha256": sha256(weights),
        "model_config": state["model_config"],
        "inference_config": config,
        "campaign_id": snapshot["run_id"],
        "checkpoint_sha256": selection["checkpoint"]["sha256"],
        "checkpoint_step": state["step"],
        "training_source_identity": campaign["identity"],
        "export_torch_version": torch.__version__,
        "selection": "fold0 development-selected checkpoint and frozen threshold",
        "known_limitations": [
            "Whole-embryo evaluation incomplete: 61 of 71 clips; other direction not evaluated.",
            "Detection caps were hit in 43 of 61 evaluated clips; promotion criteria failed.",
            "User explicitly authorized this experimental submission on 2026-09-07.",
        ],
    }
    write_json(output / "inference-model.json", manifest)
    write_json(
        output / "dataset-metadata.json",
        {
            "id": dataset_id,
            "title": "Biohub Frozen Point Detector 20260907",
            "licenses": [{"name": "CC0-1.0"}],
            "description": "Private inference weights trained on competition data; experimental.",
        },
    )
    return manifest


INFERENCE = '''import hashlib, json, signal, time
from pathlib import Path
import torch
from biohub_ct.data.paths import discover_datasets
from biohub_ct.data.zarr_io import open_zarr_volume
from biohub_ct.pipelines.learned import LearnedConfig, predict_record
from biohub_ct.pipelines.submission_pipeline import atomic_json, environment_info
from biohub_ct.submission.writer import write_submission
from biohub_ct.training.model import ModelConfig, PointDetector3D

def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def timeout_handler(signum, frame):
    raise TimeoutError('Kaggle inference runtime limit reached')

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(DEADLINE_SECONDS)
started = time.monotonic()
deadline_at = started + DEADLINE_SECONDS
manifests = list(Path('/kaggle/input').rglob('inference-model.json'))
assert len(manifests) == 1, f'Expected one model manifest, found {len(manifests)}'
manifest_path = manifests[0]
assert file_sha256(manifest_path) == EXPECTED_MANIFEST_SHA256, 'Model manifest checksum mismatch'
model_manifest = json.loads(manifest_path.read_text())
weights_path = manifest_path.parent / model_manifest['weights_file']
assert weights_path.resolve().parent == manifest_path.parent.resolve(), 'Unsafe weights path'
assert file_sha256(weights_path) == model_manifest['weights_sha256'], 'Weights checksum mismatch'
assert torch.cuda.is_available(), 'GPU required for this learned submission'
model = PointDetector3D(ModelConfig(**model_manifest['model_config'])).eval()
model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
assert all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()), 'Nonfinite weights'
model = model.cuda()
config = LearnedConfig(**model_manifest['inference_config'])
slug = 'biohub-cell-tracking-during-development'
roots = [p for p in (Path('/kaggle/input/competitions') / slug, Path('/kaggle/input') / slug)
         if (p / 'test').is_dir()]
assert len(roots) == 1, f'Expected one competition test root: {roots}'
records = discover_datasets(roots[0] / 'test', require_geff=False)
assert records, 'No test datasets found'
shapes, diagnostics = {}, []
output = Path('/kaggle/working/submission.csv')
print(json.dumps({'test_datasets': len(records), 'torch': torch.__version__,
                  'gpu': torch.cuda.get_device_name(0), 'weights_sha256': model_manifest['weights_sha256'],
                  'inference_config': model_manifest['inference_config']}), flush=True)

def graphs():
    for record in records:
        begin = time.monotonic()
        if begin >= deadline_at:
            raise TimeoutError('Inference deadline reached before next dataset')
        shapes[record.name] = open_zarr_volume(record.zarr_path, require_complete_chunks=True).shape
        graph = predict_record(model, record, config, deadline_at=deadline_at)
        row = {'dataset': record.name, 'nodes': graph.num_nodes, 'edges': graph.num_edges,
               'runtime_s': time.monotonic() - begin, **graph.inference_diagnostics}
        diagnostics.append(row)
        atomic_json(output.with_suffix('.progress.json'), {'complete': False, 'datasets': diagnostics})
        print(json.dumps({**row, 'capped_frames': len(row['capped_frames'])}), flush=True)
        yield record.name, graph

# The writer validates every graph and the complete CSV, including exact discovered coverage,
# before atomically installing submission.csv. It processes one dataset graph at a time.
write_submission(graphs(), output, expected_datasets=[r.name for r in records], shapes=shapes)
report = {'status': 'complete', 'model_manifest_sha256': EXPECTED_MANIFEST_SHA256,
          'weights_sha256': model_manifest['weights_sha256'], 'model': model_manifest,
          'source_archive_sha256': SOURCE_ARCHIVE_SHA256, 'datasets': diagnostics,
          'elapsed_s': time.monotonic() - started, 'submission_sha256': file_sha256(output),
          'torch': torch.__version__, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
          **environment_info()}
atomic_json(output.with_suffix('.manifest.json'), report)
print(json.dumps({'status': 'complete', 'datasets': len(records), 'elapsed_s': report['elapsed_s'],
                  'submission_sha256': report['submission_sha256']}), flush=True)
signal.alarm(0)
'''


def build_learned(output_dir, kernel_id, model_dir, dataset_id, deadline_seconds=32400):
    if not 1 <= deadline_seconds <= 32400:
        raise ValueError("Inference deadline must be within the nine-hour release allowance")
    model_dir = Path(model_dir)
    manifest_path = model_dir / "inference-model.json"
    model_manifest = json.loads(manifest_path.read_text())
    weights = model_dir / model_manifest["weights_file"]
    if weights.resolve().parent != model_dir.resolve():
        raise ValueError("Unsafe weights path")
    if sha256(weights) != model_manifest["weights_sha256"]:
        raise ValueError("Exported weights checksum mismatch")
    result = build(output_dir, kernel_id, deadline_seconds=deadline_seconds)
    output = Path(output_dir)
    notebook_path = output / "submission.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"][0]["source"] = [
        "# Biohub experimental learned detector\n",
        "Frozen fold0 checkpoint and threshold. Validation is incomplete and detection caps "
        "were observed. Submitted at the user's explicit request.\n",
    ]
    inference = (
        f"DEADLINE_SECONDS = {int(deadline_seconds)}\n"
        f"EXPECTED_MANIFEST_SHA256 = {sha256(manifest_path)!r}\n"
        f"SOURCE_ARCHIVE_SHA256 = {result['source_archive_sha256']!r}\n" + INFERENCE
    )
    compile(inference, "learned-inference-cell", "exec")
    notebook["cells"][-1]["source"] = inference.splitlines(keepends=True)
    write_json(notebook_path, notebook)
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        title="Biohub Frozen Learned Submission",
        enable_gpu=True,
        dataset_sources=[dataset_id],
    )
    write_json(metadata_path, metadata)
    result.update(
        model_manifest_sha256=sha256(manifest_path),
        weights_sha256=model_manifest["weights_sha256"],
        notebook_sha256=sha256(notebook_path),
        inference_config=model_manifest["inference_config"],
        model_dataset=dataset_id,
        deadline_seconds=deadline_seconds,
        known_limitations=model_manifest["known_limitations"],
    )
    write_json(output / "package-manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--snapshot", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--dataset-id", required=True)
    package = sub.add_parser("build")
    package.add_argument("--output-dir", required=True)
    package.add_argument("--kernel-id", required=True)
    package.add_argument("--model-dir", required=True)
    package.add_argument("--dataset-id", required=True)
    package.add_argument("--deadline-seconds", type=int, default=32400)
    args = parser.parse_args()
    if args.command == "export":
        result = export_model(args.checkpoint, args.snapshot, args.output_dir, args.dataset_id)
    else:
        result = build_learned(
            args.output_dir, args.kernel_id, args.model_dir, args.dataset_id, args.deadline_seconds
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
