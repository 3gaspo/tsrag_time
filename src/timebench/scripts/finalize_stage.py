"""Finalize ready runs after srun success, or interrupt the owning launch."""

import argparse
from pathlib import Path
from timebench.pipeline.runs import load_manifest, RunHandle, interrupt_launch
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('launch')
    parser.add_argument('--interrupt', action='store_true')
    args = parser.parse_args()
    if args.interrupt:
        interrupt_launch(args.root, args.launch)
        return
    for path in args.root.rglob('stage_ready.json'):
        manifest = load_manifest(path.parent)
        if manifest['status'] != 'computed' or manifest['launch']['launch_id'] != args.launch:
            continue
        ready = json.loads(path.read_text())
        RunHandle(path.parent, manifest, 'finalize').complete(ready['required_artifacts'])
        path.unlink()


if __name__ == '__main__':
    main()
