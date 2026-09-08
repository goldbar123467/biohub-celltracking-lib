"""Build the opt-in instrumented E0 public-reference notebook candidate.

The frozen public-reference builder remains the source of input verification
and of the twelve public cells.  This builder wraps those exact cell strings
with the reviewed notebook and support-script telemetry layers.  It does not
contact Kaggle or execute the notebook.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

try:
    from scripts import package_public_reference as base_package
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import package_public_reference as base_package  # type: ignore[no-redef]

from biohub_ct.campaign import (
    e0_instrumentation_launcher,
    e0_notebook_telemetry,
    e0_support_telemetry,
)

PUBLIC_CELL_COUNT = 12
PUBLIC_LAUNCH_CELL_INDEX = 4  # Zero-based index in the frozen public notebook.
PUBLIC_LAUNCH_CELL_SHA256 = (
    "a3a9827b4c0119e90111a091083658317e6b2542320d03ef69ffd031c0a64684"
)
TARGET_SUPPORT_SCRIPT = "/kaggle/working/tracking_repo/scripts/predict_unet_transformer.py"
SUPPORT_OUTPUT_DIR = "/kaggle/working/e0_support_telemetry"
SUPPORT_MODULE_NAME = "_biohub_e0_support_telemetry_packaged"
LAUNCHER_MODULE_NAME = "_biohub_e0_instrumentation_launcher_packaged"
INCOMPLETE_SENTINEL = "INCOMPLETE"
_KAGGLE_OWNER_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
_KAGGLE_SLUG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_KAGGLE_TITLE_RE = re.compile(r"[A-Za-z0-9]+(?:[ -][A-Za-z0-9]+)*\Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _kernel_title(kernel_id: str, title: str | None) -> str:
    """Return an ASCII title whose Kaggle slug is exactly ``kernel_id``'s slug.

    Kaggle CLI 2.2.4 derives the effective slug from ``title`` during push.  A
    deliberately narrow title alphabet makes that mapping local and exact:
    lowercase ASCII words separated by spaces or hyphens become the same words
    separated by hyphens.  Punctuation and Unicode transliteration are rejected
    instead of creating permissive aliases.
    """

    if not isinstance(kernel_id, str) or kernel_id.count("/") != 1:
        raise ValueError("Kernel ID must have owner/slug form")
    owner, slug = kernel_id.split("/", 1)
    if (
        _KAGGLE_OWNER_RE.fullmatch(owner) is None
        or _KAGGLE_SLUG_RE.fullmatch(slug) is None
    ):
        raise ValueError(
            "Instrumented kernel ID must contain an ASCII owner and hyphenated slug"
        )
    selected = (
        " ".join(word.capitalize() for word in slug.split("-"))
        if title is None
        else title
    )
    if (
        not isinstance(selected, str)
        or len(selected) < 5
        or not selected.isascii()
        or _KAGGLE_TITLE_RE.fullmatch(selected) is None
    ):
        raise ValueError(
            "Kernel title must be at least five ASCII letters/digits with separators"
        )
    canonical = selected.lower().replace(" ", "-")
    if canonical != slug:
        raise ValueError(
            f"Kernel title canonicalizes to {canonical!r}, expected exact slug {slug!r}"
        )
    return selected


def _module_source(module: object) -> str:
    path = Path(str(getattr(module, "__file__", "")))
    if not path.is_file():
        raise RuntimeError(f"Cannot read instrumentation module source: {path}")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    return source


def _instrumentation_config(*, launch_cell_sha256: str) -> dict[str, Any]:
    if base_package.SHA256_RE.fullmatch(launch_cell_sha256) is None:
        raise ValueError("Launch-cell SHA-256 is invalid")
    return {
        "schema_version": 1,
        "target_support_script": TARGET_SUPPORT_SCRIPT,
        "support_output_dir": SUPPORT_OUTPUT_DIR,
        "callback_after_public_cell_sha256": launch_cell_sha256,
        "expected_public_patched_support_sha256": (
            e0_support_telemetry.PUBLIC_PATCHED_SUPPORT_SHA256
        ),
        "support_runtime_module_name": e0_support_telemetry.RUNTIME_MODULE_NAME,
        "expected_launches": "2 if worker_count >= 2 and not SLICE else 1",
        "notebook_telemetry": {
            "output_dir": "/kaggle/working",
            "sample_interval_seconds": 5.0,
            "nvidia_smi_timeout_seconds": 2.0,
            "max_samples": 20_000,
            "retention_glob": "retention_guard_*.jsonl",
            "coordinate_glob": "detector_coordinates_*.jsonl",
            "support_telemetry_dirname": "e0_support_telemetry",
            "require_coordinate_artifacts": True,
            "require_submission_crosscheck": True,
        },
    }


def _install_interceptor_source(
    *,
    config: Mapping[str, Any],
    support_source: str,
    launcher_source: str,
) -> str:
    config_json = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    support_hash = _sha256_text(support_source)
    launcher_hash = _sha256_text(launcher_source)
    return f'''# E0 support instrumentation: install immediately before public cells.
import atexit as _e0_atexit
import hashlib as _e0_hashlib
import json as _e0_json
import os as _e0_os
import sys as _e0_sys
import types as _e0_types
from pathlib import Path as _E0InstrumentationPath

_E0_INSTRUMENTATION_CONFIG = _e0_json.loads({config_json!r})

def _e0_load_instrumentation_module(name, source, expected_sha256, filename):
    if _e0_hashlib.sha256(source.encode("utf-8")).hexdigest() != expected_sha256:
        raise RuntimeError("embedded instrumentation source hash mismatch: " + name)
    if name in _e0_sys.modules:
        raise RuntimeError("embedded instrumentation module is already registered: " + name)
    module = _e0_types.ModuleType(name)
    module.__file__ = filename
    _e0_sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except BaseException:
        _e0_sys.modules.pop(name, None)
        raise
    return module

_E0_SUPPORT_MODULE_SOURCE = {support_source!r}
_E0_LAUNCHER_MODULE_SOURCE = {launcher_source!r}
_E0_SUPPORT_MODULE = _e0_load_instrumentation_module(
    {SUPPORT_MODULE_NAME!r},
    _E0_SUPPORT_MODULE_SOURCE,
    {support_hash!r},
    "<embedded-e0-support-telemetry>",
)
_E0_LAUNCHER_MODULE = _e0_load_instrumentation_module(
    {LAUNCHER_MODULE_NAME!r},
    _E0_LAUNCHER_MODULE_SOURCE,
    {launcher_hash!r},
    "<embedded-e0-instrumentation-launcher>",
)

def _e0_pinned_support_patcher(source):
    return _E0_SUPPORT_MODULE.patch_support_source(
        source,
        expected_public_patched_sha256=(
            _E0_INSTRUMENTATION_CONFIG["expected_public_patched_support_sha256"]
        ),
    )

_E0_SUPPORT_INTERCEPTOR = _E0_LAUNCHER_MODULE.SupportScriptInterceptor(
    _E0_INSTRUMENTATION_CONFIG["target_support_script"],
    expected_public_sha256=(
        _E0_INSTRUMENTATION_CONFIG["expected_public_patched_support_sha256"]
    ),
    patcher=_e0_pinned_support_patcher,
    helper_source=_E0_SUPPORT_MODULE.runtime_helper_source(),
    runtime_module_name=_E0_INSTRUMENTATION_CONFIG["support_runtime_module_name"],
    output_dir=_E0_INSTRUMENTATION_CONFIG["support_output_dir"],
)
_E0_SUPPORT_INTERCEPTOR.install()
_E0_LAUNCH_CALLBACK_STATE = {{
    "schema_version": 1,
    "status": "PENDING",
    "callback_after_public_cell_sha256": (
        _E0_INSTRUMENTATION_CONFIG["callback_after_public_cell_sha256"]
    ),
}}
_E0_LAUNCH_CALLBACK_STATUS_PATH = (
    _E0InstrumentationPath(_E0_INSTRUMENTATION_CONFIG["support_output_dir"])
    / "launcher-callback-status.json"
)

def _e0_persist_launch_callback_state():
    payload = (_e0_json.dumps(
        _E0_LAUNCH_CALLBACK_STATE,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\\n").encode("utf-8")
    temporary = _E0_LAUNCH_CALLBACK_STATUS_PATH.with_name(
        "." + _E0_LAUNCH_CALLBACK_STATUS_PATH.name + ".tmp"
    )
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        _e0_os.fsync(stream.fileno())
    _e0_os.replace(temporary, _E0_LAUNCH_CALLBACK_STATUS_PATH)

def _e0_restore_after_public_launch_cell(result):
    info = getattr(result, "info", None)
    raw_cell = getattr(info, "raw_cell", None)
    if not isinstance(raw_cell, str):
        return
    actual_sha256 = _e0_hashlib.sha256(raw_cell.encode("utf-8")).hexdigest()
    if actual_sha256 != _E0_INSTRUMENTATION_CONFIG["callback_after_public_cell_sha256"]:
        return
    if _E0_LAUNCH_CALLBACK_STATE["status"] != "PENDING":
        _E0_LAUNCH_CALLBACK_STATE.update({{
            "status": "ERROR",
            "error_type": "RuntimeError",
            "error": "launch-cell callback executed more than once",
        }})
        _e0_persist_launch_callback_state()
        raise RuntimeError(_E0_LAUNCH_CALLBACK_STATE["error"])
    try:
        expected_launches = (
            2
            if int(globals().get("worker_count", 0)) >= 2
            and not bool(globals().get("SLICE"))
            else 1
        )
        observation = _E0_SUPPORT_INTERCEPTOR.restore_launcher(
            expected_launches=expected_launches
        )
        execution_error = (
            getattr(result, "error_before_exec", None)
            or getattr(result, "error_in_exec", None)
        )
        if execution_error is not None:
            raise RuntimeError(
                "public launch cell reported an execution error: "
                + type(execution_error).__name__
            )
        _E0_LAUNCH_CALLBACK_STATE.update({{
            "status": "PASS",
            "launch_count": observation["launch_count"],
            "expected_launch_count": observation["expected_launch_count"],
        }})
    except BaseException as error:
        _E0_LAUNCH_CALLBACK_STATE.update({{
            "status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
        }})
        _e0_persist_launch_callback_state()
        raise
    _e0_persist_launch_callback_state()

_E0_INSTRUMENTATION_SHELL = get_ipython()
_E0_INSTRUMENTATION_SHELL.events.register(
    "post_run_cell", _e0_restore_after_public_launch_cell
)

def _e0_instrumentation_atexit_cleanup():
    try:
        _E0_INSTRUMENTATION_SHELL.events.unregister(
            "post_run_cell", _e0_restore_after_public_launch_cell
        )
    except (KeyError, ValueError):
        pass
    finally:
        _E0_SUPPORT_INTERCEPTOR.close()

_e0_atexit.register(_e0_instrumentation_atexit_cleanup)
_e0_persist_launch_callback_state()
print("E0_SUPPORT_INTERCEPTOR_INSTALLED", {{
    "target": _E0_INSTRUMENTATION_CONFIG["target_support_script"],
    "expected_public_sha256": (
        _E0_INSTRUMENTATION_CONFIG["expected_public_patched_support_sha256"]
    ),
}}, flush=True)
'''


def _close_interceptor_source() -> str:
    return '''# E0 support instrumentation: require callback success and always clean up.
_e0_close_problems = []
_e0_close_state = globals().get("_E0_LAUNCH_CALLBACK_STATE")
if not isinstance(_e0_close_state, dict):
    _e0_close_problems.append("launch callback state is missing")
elif _e0_close_state.get("status") != "PASS":
    _e0_close_problems.append(
        "launch callback did not pass: " + repr(_e0_close_state)
    )

_e0_close_shell = globals().get("_E0_INSTRUMENTATION_SHELL")
_e0_close_callback = globals().get("_e0_restore_after_public_launch_cell")
try:
    if _e0_close_shell is None or _e0_close_callback is None:
        _e0_close_problems.append("launch callback registration is missing")
    else:
        _e0_close_shell.events.unregister("post_run_cell", _e0_close_callback)
except BaseException as _e0_unregister_error:
    _e0_close_problems.append(
        "callback unregister failed: " + type(_e0_unregister_error).__name__
    )
finally:
    try:
        _e0_hook = globals().get("_e0_instrumentation_atexit_cleanup")
        if _e0_hook is None:
            _e0_close_problems.append("atexit cleanup hook is missing")
        else:
            _e0_atexit.unregister(_e0_hook)
    except BaseException as _e0_atexit_error:
        _e0_close_problems.append(
            "atexit unregister failed: " + type(_e0_atexit_error).__name__
        )
    finally:
        try:
            _e0_interceptor = globals().get("_E0_SUPPORT_INTERCEPTOR")
            if _e0_interceptor is None:
                _e0_close_problems.append("support interceptor is missing")
            else:
                _e0_interceptor.close()
        except BaseException as _e0_interceptor_close_error:
            _e0_close_problems.append(
                "interceptor close failed: "
                + type(_e0_interceptor_close_error).__name__
            )

if isinstance(_e0_close_state, dict):
    _e0_close_state["cleanup_status"] = (
        "PASS" if not _e0_close_problems else "ERROR"
    )
    _e0_close_state["cleanup_errors"] = list(_e0_close_problems)
    try:
        _e0_persist_launch_callback_state()
    except BaseException as _e0_status_write_error:
        _e0_close_problems.append(
            "callback status write failed: " + type(_e0_status_write_error).__name__
        )
if _e0_close_problems:
    raise RuntimeError({"e0_support_instrumentation_close": _e0_close_problems})
print("E0_SUPPORT_INTERCEPTOR_CLOSED", _e0_close_state, flush=True)
'''


def _cell_source(cell: Mapping[str, Any]) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


def build_instrumented_package(
    source_notebook: Path,
    archives: Mapping[str, Path],
    output_dir: Path,
    kernel_id: str,
    *,
    pin: Mapping[str, Any] = base_package.PINNED_REFERENCE,
    launch_cell_sha256: str = PUBLIC_LAUNCH_CELL_SHA256,
    title: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic candidate, leaving INCOMPLETE on any failure."""

    kernel_title = _kernel_title(kernel_id, title)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sentinel = output_dir / INCOMPLETE_SENTINEL
    sentinel.write_text(
        "Instrumented E0 package construction did not complete.\n",
        encoding="utf-8",
        newline="\n",
    )

    with tempfile.TemporaryDirectory(prefix=".base-package-", dir=output_dir) as temporary:
        base_output = Path(temporary)
        base_manifest = base_package.build_package(
            source_notebook,
            archives,
            base_output,
            kernel_id,
            pin=pin,
        )
        base_notebook = json.loads(
            (base_output / "submission.ipynb").read_text(encoding="utf-8")
        )
        base_lock = json.loads(
            (base_output / "artifact-lock.json").read_text(encoding="utf-8")
        )
        kernel_metadata = json.loads(
            (base_output / "kernel-metadata.json").read_text(encoding="utf-8")
        )

    public_cells = copy.deepcopy(base_notebook["cells"][1:-1])
    if len(public_cells) != PUBLIC_CELL_COUNT:
        raise RuntimeError(
            f"Expected {PUBLIC_CELL_COUNT} frozen public cells, got {len(public_cells)}"
        )
    public_sources = [_cell_source(cell) for cell in public_cells]
    public_hashes = [_sha256_text(source) for source in public_sources]
    if public_hashes[PUBLIC_LAUNCH_CELL_INDEX] != launch_cell_sha256:
        raise RuntimeError(
            "Frozen public launch-cell SHA-256 differs from the reviewed callback target"
        )
    if public_hashes.count(launch_cell_sha256) != 1:
        raise RuntimeError("Reviewed public launch-cell SHA-256 must occur exactly once")

    support_source = _module_source(e0_support_telemetry)
    launcher_source = _module_source(e0_instrumentation_launcher)
    notebook_telemetry_source = _module_source(e0_notebook_telemetry)
    builder_source = Path(__file__).read_text(encoding="utf-8")
    support_runtime_source = e0_support_telemetry.runtime_helper_source()
    instrumentation_config = _instrumentation_config(
        launch_cell_sha256=launch_cell_sha256
    )
    config_sha256 = hashlib.sha256(
        base_package.canonical_json_bytes(instrumentation_config)
    ).hexdigest()

    artifact_lock = copy.deepcopy(base_lock)
    artifact_lock.pop("release_digest", None)
    artifact_lock["instrumentation"] = {
        "schema_version": 1,
        "config": instrumentation_config,
        "config_sha256": config_sha256,
        "source_sha256": {
            "e0_support_telemetry.py": _sha256_text(support_source),
            "e0_support_runtime_helper.py": _sha256_text(support_runtime_source),
            "e0_instrumentation_launcher.py": _sha256_text(launcher_source),
            "e0_notebook_telemetry.py": _sha256_text(notebook_telemetry_source),
            "package_instrumented_public_reference.py": _sha256_text(builder_source),
        },
    }
    artifact_lock["release_digest"] = hashlib.sha256(
        base_package.canonical_json_bytes(artifact_lock)
    ).hexdigest()

    install_source = _install_interceptor_source(
        config=instrumentation_config,
        support_source=support_source,
        launcher_source=launcher_source,
    )
    integrity_source = base_package._preflight_source(artifact_lock)
    validation_source = base_package._release_validation_source(artifact_lock)
    close_source = _close_interceptor_source()
    allowed_auxiliary = tuple(
        _sha256_text(source)
        for source in (
            install_source,
            integrity_source,
            validation_source,
            close_source,
        )
    )
    telemetry_config = e0_notebook_telemetry.NotebookTelemetryConfig(
        expected_public_cell_sha256=tuple(public_hashes),
        allowed_auxiliary_cell_sha256=allowed_auxiliary,
        **instrumentation_config["notebook_telemetry"],
    )
    telemetry_sources = e0_notebook_telemetry.build_notebook_telemetry_sources(
        telemetry_config
    )

    packaged = copy.deepcopy(base_notebook)
    packaged["cells"] = [
        base_package._code_cell(
            telemetry_sources.prepended_source, "e0-telemetry-bootstrap"
        ),
        base_package._code_cell(install_source, "e0-support-interceptor-install"),
        base_package._code_cell(integrity_source, "e0-input-integrity"),
        *public_cells,
        base_package._code_cell(validation_source, "e0-release-validation"),
        base_package._code_cell(close_source, "e0-support-interceptor-close"),
        base_package._code_cell(telemetry_sources.final_source, "e0-telemetry-finalizer"),
    ]
    if [_cell_source(cell) for cell in packaged["cells"][3:15]] != public_sources:
        raise RuntimeError("Frozen public cell sources changed during instrumentation")
    for cell in packaged["cells"]:
        if cell.get("cell_type") == "code":
            compile(_cell_source(cell), "instrumented-packaged-notebook-cell", "exec")

    packaged_metadata = packaged.setdefault("metadata", {})
    if not isinstance(packaged_metadata.get("biohub_e0"), dict):
        raise TypeError("Base package is missing biohub_e0 notebook metadata")
    packaged_metadata["biohub_e0"]["release_digest"] = artifact_lock["release_digest"]
    packaged_metadata["biohub_e0_instrumented"] = {
        "release_digest": artifact_lock["release_digest"],
        "base_release_digest": base_manifest["release_digest"],
        "algorithm_cells_preserved": PUBLIC_CELL_COUNT,
        "instrumentation_config_sha256": config_sha256,
        "candidate_status": "not_executed",
    }
    notebook_path = output_dir / "submission.ipynb"
    base_package.write_json(notebook_path, packaged)

    kernel_metadata["title"] = kernel_title
    base_package.write_json(output_dir / "kernel-metadata.json", kernel_metadata)
    base_package.write_json(output_dir / "artifact-lock.json", artifact_lock)

    cell_layout = [
        "e0-telemetry-bootstrap",
        "e0-support-interceptor-install",
        "e0-input-integrity",
        *(["public"] * PUBLIC_CELL_COUNT),
        "e0-release-validation",
        "e0-support-interceptor-close",
        "e0-telemetry-finalizer",
    ]
    manifest = copy.deepcopy(base_manifest)
    manifest.update(
        {
            "status": "ready_for_root_review_not_launched",
            "candidate_kind": "instrumented_public_reference",
            "instrumentation_status": "not_executed",
            "release_digest": artifact_lock["release_digest"],
            "base_release_digest": base_manifest["release_digest"],
            "packaged_notebook": {
                "path": notebook_path.name,
                "sha256": base_package.sha256_file(notebook_path),
                "cell_count": len(packaged["cells"]),
                "algorithm_cell_positions": [3, 14],
                "algorithm_cell_source_sha256": public_hashes,
                "instrumentation": cell_layout,
            },
            "kernel_metadata_sha256": base_package.sha256_file(
                output_dir / "kernel-metadata.json"
            ),
            "artifact_lock_sha256": base_package.sha256_file(
                output_dir / "artifact-lock.json"
            ),
            "instrumentation": {
                "config_sha256": config_sha256,
                "source_sha256": artifact_lock["instrumentation"]["source_sha256"],
                "generated_cell_sha256": {
                    "telemetry_bootstrap": telemetry_sources.prepended_source_sha256,
                    "interceptor_install": allowed_auxiliary[0],
                    "input_integrity": allowed_auxiliary[1],
                    "base_validation": allowed_auxiliary[2],
                    "interceptor_close": allowed_auxiliary[3],
                    "telemetry_finalizer": telemetry_sources.final_source_sha256,
                },
                "allowed_auxiliary_cell_sha256": list(allowed_auxiliary),
            },
            "algorithm_changes": [],
            "packaging_changes": [
                "wrap the exact 12 public cells with additive notebook telemetry",
                "intercept the exact public support-script launch and apply the pinned telemetry patch",
                "restore the subprocess launcher after the reviewed public launch cell",
                "retain the base input-integrity and release-validation logic under the new lock",
            ],
            "known_limitations": [
                *base_manifest.get("known_limitations", []),
                "Instrumented candidate has not passed a Kaggle execution or release-admission comparison.",
            ],
        }
    )
    base_package.write_json(output_dir / "package-manifest.json", manifest)
    sentinel.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument(
        "--archive", action="append", default=[], metavar="REF=PATH", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument(
        "--title",
        help="optional narrow ASCII title that canonicalizes exactly to --kernel-id's slug",
    )
    args = parser.parse_args()
    manifest = build_instrumented_package(
        args.source_notebook,
        base_package.parse_archive_arguments(args.archive),
        args.output_dir,
        args.kernel_id,
        title=args.title,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
