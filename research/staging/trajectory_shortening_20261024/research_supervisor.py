"""Finite Agent-side read-only research refresh, with separate output leaves.

Every five hours refresh length diagnostics for seven days or until the matrix
has no pending/running cells. It never launches GPU work or changes run states.
The control-terminal Issue reporter reads research-status.json independently.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--cycles', type=int, default=34)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    audit = Path(__file__).with_name('audit_recordings.py')
    expected = hashlib.sha256(audit.read_bytes()).hexdigest()
    with (args.root / 'research-supervisor.lock').open('a') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.nice(15)
        for cycle in range(args.cycles):
            assert hashlib.sha256(audit.read_bytes()).hexdigest() == expected
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            out = args.root / 'length-refresh' / stamp
            out.parent.mkdir(exist_ok=True)
            with (args.root / ('length-' + stamp + '.log')).open('wb') as log:
                result = subprocess.run([sys.executable, str(audit), '--matrix', str(args.matrix),
                                         '--out', str(out)], stdout=log, stderr=subprocess.STDOUT, timeout=240)
            status_path = args.root / 'research-status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status['supervisor'] = dict(pid=os.getpid(), cycle=cycle, maximum_cycles=args.cycles,
                                        updated_utc=stamp, last_exit_code=result.returncode,
                                        interval_hours=5, launches_training=False)
            if not result.returncode:
                value = json.loads((out / 'summary.json').read_text())
                human = value['human_xyz_learned_trainable']['plr']
                status['length'] = dict(observed_utc=value['finished_utc'],
                    completed=value['counts'].get('completed'), rejected_n=len(value['rejected']),
                    human_cells=human['n'], human_fraction_below_one=human['fraction_below_one'],
                    canonical_cluster=value['canonical_t200_avg_velocity'],
                    summary_path=str(out / 'summary.json'), audit_source_sha256=expected,
                    formal_full_chain_admission=False)
            save(status_path, status)
            if not result.returncode and value['counts'].get('qualified', 0) == 0 and value['counts'].get('running', 0) == 0:
                return
            if cycle + 1 < args.cycles:
                time.sleep(5 * 3600)


if __name__ == '__main__':
    main()
