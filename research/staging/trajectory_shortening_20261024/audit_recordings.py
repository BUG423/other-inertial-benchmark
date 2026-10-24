"""Read completed result metadata on the Agent; never load weights or raw IMU.

Inference is conditional on observed profiles/sources. Subject groups are the
bootstrap units within each fixed source; seeds and repeated profiles are NOT
independent samples. This diagnostic is not the formal full-chain admission.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics as stats

METRICS = ('plr', 'plr_oracle', 'oracle_ratio', 'speed_ratio', 'along_bias',
           'dir_err_mean', 'ate', 'ate_oracle', 'rte', 't_rte_10s')
HUMAN = {'ridi', 'ronin', 'idol', 'rnin', 'tlio', 'oxiod', 'imunet', 'pedlocdata'}


def number(x):
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def describe(values):
    values = [x for x in values if x is not None]
    return dict(n=len(values), mean=stats.fmean(values) if values else None,
                median=stats.median(values) if values else None,
                fraction_below_one=stats.fmean(x < 1 for x in values) if values else None,
                fraction_below_095=stats.fmean(x < .95 for x in values) if values else None)


def bounded(path, limit=32 * 1024 ** 2):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Metadata size bound exceeded: ' + str(path))
    return data


def bootstrap(rows, repeats=2000):
    # First collapse repeated seeds/profiles for the same recording, then
    # recordings within subject. Each subject has one vote within its source.
    recordings = defaultdict(list)
    for row in rows:
        recordings[(row['dataset'], row['group_id'], row['sequence_id'])].append(row)
    subjects = defaultdict(list)
    for (dataset, group, sid), repeated in recordings.items():
        item = {}
        for key in ('plr', 'plr_oracle', 'speed_ratio'):
            v = [x[key] for x in repeated if x[key] is not None]
            item[key] = stats.fmean(v) if v else None
        subjects[(dataset, group)].append(item)
    sources = defaultdict(list)
    for (dataset, group), records in subjects.items():
        item = {}
        for key in ('plr', 'plr_oracle', 'speed_ratio'):
            v = [x[key] for x in records if x[key] is not None]
            if v:
                item[key] = stats.fmean(v)
                item[key + '_short_fraction'] = stats.fmean(x < .95 for x in v)
        sources[dataset].append(item)
    rng = random.Random(741092)
    result = {}
    for key in ('plr', 'plr_oracle', 'speed_ratio', 'plr_short_fraction', 'plr_oracle_short_fraction'):
        groups = [[x[key] for x in group if key in x] for group in sources.values()]
        groups = [g for g in groups if g]
        if not groups:
            continue
        distribution = sorted(stats.fmean(stats.fmean(rng.choices(g, k=len(g))) for g in groups)
                              for _ in range(repeats))
        result[key] = dict(estimate=stats.fmean(stats.fmean(g) for g in groups),
                           conditional_subject_bootstrap_95=[distribution[int(.025 * repeats)],
                                                             distribution[int(.975 * repeats)]])
    return dict(subject_groups=len(subjects), recordings=len(recordings),
                groups_by_source={k: len(v) for k, v in sources.items()}, metrics=result,
                limitation='Fixed observed algorithms/sources; not an algorithm-population CI. '
                           'Subject IDs follow existing group_id; absent IDs fall back to recording.')


def audit(matrix, out, repeats):
    start = datetime.now(timezone.utc).isoformat()
    cov_bytes = bounded(matrix / 'coverage.json')
    coverage = json.loads(cov_bytes)
    rows = coverage['coverage']
    assert len(rows) == 2067
    seeds = set(coverage['seeds'])
    counts, rejected, cells, recordings, artifacts = Counter(), [], [], [], []
    for original in rows:
        identifier = original['id']
        if not identifier.isalnum():
            raise ValueError('Unsafe row ID')
        state_path = matrix / 'run-states' / (identifier + '.json')
        state_bytes = bounded(state_path, 2 * 1024 ** 2) if state_path.exists() else b'{}'
        state = json.loads(state_bytes)
        status = state.get('state', original.get('state'))
        counts[status] += 1
        if status != 'completed':
            continue
        run = matrix / 'runs' / identifier / original['profile'] / original['dataset'] / f"seed{original['seed']}" / 'test'
        if not run.resolve().is_relative_to(matrix.resolve()):
            raise ValueError('Unsafe result path')
        try:
            mb, cb = bounded(run / 'metrics.json'), bounded(run / 'sequences.csv')
            meta = json.loads(mb)
            if (meta['model'] != original['profile'] or meta['dataset'] != original['dataset']
                    or meta['seed'] != original['seed'] or meta['split'] != 'test'
                    or meta['dataset_fingerprint'] != original['dataset_fingerprint']
                    or int(meta['metric_dims_effective']) != original['dims']):
                raise ValueError('Result identity mismatch')
            records = list(csv.DictReader(io.StringIO(cb.decode())))
            if len({r['sequence_id'] for r in records}) != len(records):
                raise ValueError('Duplicate sequence')
            cell = {k: original.get(k) for k in ('id', 'profile', 'architecture', 'dataset', 'seed', 'dims', 'classic', 'input_track', 'trainable')}
            cell['target'] = original.get('scientific_input_spec', {}).get('target')
            cell['window'] = original.get('scientific_input_spec', {}).get('window')
            cell.update({k: number(meta['metrics'].get(k)) for k in METRICS})
            cell_records = []
            for record in records:
                if record.get('skipped'):
                    continue
                if int(record['metric_dims_effective']) != original['dims']:
                    raise ValueError('Sequence dimension mismatch')
                values = {key: number(record.get(key)) for key in METRICS}
                if values['plr'] is None:
                    continue
                dense = number(record.get('plr_dense'))
                if all(x is not None for x in (dense, values['plr_oracle'], values['oracle_ratio'])):
                    if not math.isclose(dense, values['plr_oracle'] * values['oracle_ratio'], rel_tol=1e-6, abs_tol=1e-8):
                        raise ValueError('Oracle decomposition mismatch')
                cell_records.append({**cell, **values, 'sequence_id': record['sequence_id'],
                                   'group_id': record.get('group_id') or record['sequence_id'],
                                   'valid_segments': number(record.get('valid_segments')),
                                   'gap_sensitive': values['ate_oracle'] is not None and values['ate'] is not None
                                                    and values['ate_oracle'] > values['ate']})
            cells.append(cell)
            recordings.extend(cell_records)
            artifacts.append(dict(id=identifier, state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                                  metrics_sha256=hashlib.sha256(mb).hexdigest(),
                                  csv_sha256=hashlib.sha256(cb).hexdigest()))
        except (OSError, KeyError, ValueError) as exc:
            rejected.append(dict(id=identifier, error=str(exc)))
    primary = [c for c in cells if c['dataset'] in HUMAN and c['dims'] == 3 and not c['classic'] and c['trainable']]
    groups = defaultdict(list)
    for c in primary:
        groups[(c['profile'], c['dataset'])].append(c)
    complete = {k for k, v in groups.items() if {c['seed'] for c in v} == seeds}
    strict = [r for r in recordings if (r['profile'], r['dataset']) in complete
              and r['dims'] == 3 and not r['classic'] and r['trainable']]
    by_source = {}
    for dataset in sorted(HUMAN):
        subset = [c for c in primary if c['dataset'] == dataset]
        by_source[dataset] = {key: describe([c[key] for c in subset]) for key in ('plr', 'plr_oracle', 'speed_ratio', 'dir_err_mean')}
    counterexamples = sorted((c for c in primary if c['plr'] is not None and c['plr'] >= 1),
                             key=lambda c: c['plr'], reverse=True)
    summary = dict(schema='ib.length-diagnostic.subject-cluster.v1', started_utc=start,
                   finished_utc=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
                   coverage_sha256=hashlib.sha256(cov_bytes).hexdigest(),
                   primary_test_cells=len(cells), rejected=rejected,
                   all_test_cells=describe([c['plr'] for c in cells]),
                   human_xyz_learned_trainable={k: describe([c[k] for c in primary]) for k in METRICS},
                   complete_three_seed_profile_dataset_groups=len(complete),
                   complete_seed_sequence_evaluations=len(strict),
                   subject_bootstrap=bootstrap(strict, repeats),
                   excluding_gap_sensitive=bootstrap([r for r in strict if not r['gap_sensitive'] and r['valid_segments'] == 1], repeats),
                   by_input_track={track: {k: describe([c[k] for c in primary if c['input_track'] == track])
                                           for k in ('plr', 'plr_oracle', 'speed_ratio')}
                                   for track in sorted({c['input_track'] for c in primary})},
                   canonical_t200_avg_velocity=bootstrap([r for r in strict if r['window'] == 200
                                                          and r['target'] == 'avg_velocity'
                                                          and r['input_track'] == 'reference_oracle'], repeats),
                   by_source=by_source, counterexamples=counterexamples[:20],
                   full_chain_admission=False, claims_proven=[],
                   caveats=['Primary test only; test-only transfers separated from human training.',
                            'Completed cell states are not proof of full-chain scientific validity.',
                            'Pending profiles can change estimates; no test-based tuning or gain fit.',
                            'No causal explanation from aggregated length ratios.'])
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('summary.json', summary), ('input-hashes.json', artifacts)]:
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    assert bounded(matrix / 'coverage.json') == cov_bytes
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    audit(a.matrix, a.out, a.bootstrap)
