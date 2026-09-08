"""Receive or execute an already admitted campaign worker ticket on Linux."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biohub_ct.campaign.worker_ticket import dispatch_ticket, execute_ticket, validate_ticket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dispatch", type=Path)
    group.add_argument("--execute", type=Path)
    group.add_argument("--preflight", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.preflight:
        ticket = json.loads(args.preflight.read_text())
        result = {"preflight": "PASS", "run_spec_sha256": ticket["run_spec_sha256"],
                  "intent_id": ticket["intent_id"], "fencing_token": ticket["fencing_token"],
                  "files": validate_ticket(ticket, root)}
    elif args.dispatch:
        result = dispatch_ticket(args.dispatch, root, supervisor_script=Path(__file__).resolve())
    else:
        result = execute_ticket(args.execute, root)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
