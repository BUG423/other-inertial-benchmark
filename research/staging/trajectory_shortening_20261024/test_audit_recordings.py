"""Agent-side mathematical and rejection tests for the diagnostic."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from audit_recordings import audit, bootstrap


class AuditTests(unittest.TestCase):
    def test_repeated_seed_rows_do_not_create_new_subjects(self):
        row = dict(dataset='ridi', group_id='person1', sequence_id='record1',
                   plr=.9, plr_oracle=.92, speed_ratio=.95)
        a = bootstrap([row], 100)
        b = bootstrap([row] * 30, 100)
        self.assertEqual(a, b)
        self.assertEqual(a['subject_groups'], 1)
        self.assertAlmostEqual(a['metrics']['plr']['estimate'], .9)

    def test_recordings_of_same_subject_remain_one_bootstrap_unit(self):
        rows = [dict(dataset='ridi', group_id='person1', sequence_id=str(i),
                     plr=.8 + .2 * i, plr_oracle=.9, speed_ratio=.95) for i in range(2)]
        result = bootstrap(rows, 100)
        self.assertEqual(result['subject_groups'], 1)
        self.assertEqual(result['recordings'], 2)

    def test_invalid_decomposition_cannot_leave_partial_accepted_rows(self):
        with tempfile.TemporaryDirectory() as d:
            matrix = Path(d) / 'matrix'
            (matrix / 'run-states').mkdir(parents=True)
            first = dict(id='a1', profile='p', architecture='p', dataset='ridi', seed=1,
                         dims=3, classic=False, input_track='reference_oracle', trainable=True,
                         dataset_fingerprint='fingerprint', state='completed')
            rows = [first] + [dict(id=f'b{i}', state='qualified') for i in range(2066)]
            (matrix / 'coverage.json').write_text(json.dumps(dict(coverage=rows, seeds=[1, 2, 3])))
            run = matrix / 'runs/a1/p/ridi/seed1/test'
            run.mkdir(parents=True)
            meta = dict(model='p', dataset='ridi', seed=1, split='test', metric_dims_effective=3,
                        dataset_fingerprint='fingerprint', metrics={'plr': .9})
            (run / 'metrics.json').write_text(json.dumps(meta))
            (run / 'sequences.csv').write_text(
                'sequence_id,metric_dims_effective,plr,plr_dense,plr_oracle,oracle_ratio\n'
                'valid,3,.9,.81,.9,.9\ninvalid,3,.9,100,.9,.9\n')
            with contextlib.redirect_stdout(io.StringIO()):
                audit(matrix, Path(d) / 'out', 100)
            result = json.loads((Path(d) / 'out/summary.json').read_text())
            self.assertEqual(result['primary_test_cells'], 0)
            self.assertEqual(len(result['rejected']), 1)


if __name__ == '__main__':
    unittest.main()
