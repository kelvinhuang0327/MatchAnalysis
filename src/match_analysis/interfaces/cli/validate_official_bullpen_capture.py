"""Offline official capture validation, using the existing research adapter."""
import argparse
import json
from pathlib import Path

from ...application.use_cases.adapt_legacy_bullpen_evidence import adapt_legacy_bullpen_evidence, load_json
from ...application.use_cases.convert_official_bullpen_capture import convert_official_bullpen_capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feed", "capture", "schedule", "target", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    result = convert_official_bullpen_capture(args.feed.read_bytes(), load_json(args.capture.read_bytes()),
                                             load_json(args.schedule.read_bytes()))
    admission = adapt_legacy_bullpen_evidence(result["source_bytes"], source_format="phase63-jsonl",
                                              target=load_json(args.target.read_bytes()),
                                              evidence=result["capture_evidence"])
    envelope = {"diagnostics": result["diagnostics"], "legacy_jsonl": result["source_bytes"].decode(),
                "capture_evidence": result["capture_evidence"], "admission": admission}
    with args.output.open("x") as handle:
        handle.write(json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
