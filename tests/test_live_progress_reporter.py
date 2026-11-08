"""Reporter privacy and command quoting without a network connection."""

import ast
import importlib.util
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools/ops_v1/live_progress_reporter.py'
SPEC = importlib.util.spec_from_file_location('live_progress_reporter', SOURCE)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class ReporterPrivacyTests(unittest.TestCase):
    def test_issue_target_can_be_replaced_without_changing_source(self):
        with patch.dict(os.environ, {'IPB_REPORT_ISSUE': '42'}):
            configured = importlib.util.module_from_spec(SPEC)
            SPEC.loader.exec_module(configured)
        self.assertEqual(configured.ISSUE, '42')

    def test_public_report_omits_private_runtime_fields(self):
        body, _ = reporter.render({
            'coverage_sha256': 'f' * 64,
            'hostname': 'private-host-sentinel',
            'matching_processes': ['private-command-token-sentinel'],
            'matching_process_count': 2,
            'cause': {'canonical': {'speed_ratio': {'median': 0.9}},
                      'leaf': '/private/research-sentinel'},
        })
        for private in ('private-host-sentinel', 'private-command-token-sentinel',
                        '/private/research-sentinel'):
            self.assertNotIn(private, body)
        self.assertIn('benchmark 进程数：2', body)
        self.assertNotRegex(body, r'[0-9a-fA-F]{64}')

    def test_private_paths_are_data_in_quoted_remote_command(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'profile.json'
            profile.write_text('{}')
            profile.chmod(0o600)
            matrix = "/private/space and 'quote/matrix"
            research = '/private/$(literal)/research'
            result = {'exitCode': 0, 'cleanupConfirmed': True,
                      'output': json.dumps({'full_scope': 2067})}
            with patch.multiple(reporter, PAIRING_PROFILE=profile,
                                REMOTE_AGENT_MATRIX_ROOT=matrix,
                                REMOTE_AGENT_RESEARCH_ROOT=research,
                                REMOTE_AGENT_PYTHON='/private/python path/python3'), \
                    patch.object(reporter, 'run', return_value=SimpleNamespace(
                        returncode=0, stdout=json.dumps(result))) as run:
                self.assertEqual(reporter.fetch_remote()['full_scope'], 2067)
            command = run.call_args.args[0][-1]
            python, flag, script = shlex.split(command)
            self.assertEqual(python, '/private/python path/python3')
            self.assertEqual(flag, '-c')
            tree = ast.parse(script)
            strings = {node.value for node in ast.walk(tree)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            self.assertIn(matrix, strings)
            self.assertIn(research, strings)
            self.assertNotIn('__REMOTE_AGENT_MATRIX_ROOT__', script)

    def test_undefined_sensitivity_does_not_break_a_status_report(self):
        body, _ = reporter.render({'cause': {'canonical': {
            'direction_sensitivity_descriptive': {
                'all': {'n': 3, 'correlation': -0.925},
                'direction_lt45': {'n': 2, 'correlation': None},
                'within_profile_dataset': {'groups': 0, 'correlation': None},
            },
        }}})
        self.assertIn('r=-0.9250', body)
        self.assertIn('r=unknown', body)
        self.assertIn('描述性结果', body)

    def test_capacity_report_keeps_scope_and_omits_private_bindings(self):
        evidence = SOURCE.parents[2] / 'research/staging/trajectory_shortening_20261024/capacity_observation_20261108.json'
        capacity = json.loads(evidence.read_text())
        capacity.update(leaf='/private/capacity-sentinel', source_sha256='a' * 64)
        body, _ = reporter.render({'cause': {'capacity_probe': capacity}})
        self.assertIn('单录制、16 窗', body)
        self.assertIn('0.9836', body)
        self.assertIn('不代表轨迹 PLR', body)
        self.assertNotIn('/private/capacity-sentinel', body)
        self.assertNotRegex(body, r'[0-9a-fA-F]{64}')
        capacity['steps_per_arm'] = 1
        body, _ = reporter.render({'cause': {'capacity_probe': capacity}})
        self.assertNotIn('CPU 固定训练窗对照', body)
        capacity['steps_per_arm'] = 192
        capacity['complete'] = False
        body, _ = reporter.render({'cause': {'capacity_probe': capacity}})
        self.assertNotIn('CPU 固定训练窗对照', body)

    def test_checkpoint_counterevidence_is_included_without_paths_or_hashes(self):
        evidence = SOURCE.parents[2] / 'research/staging/trajectory_shortening_20261024/checkpoint_bn_observation_20261108.json'
        probe = json.loads(evidence.read_text())
        probe.update(leaf='/private/checkpoint-sentinel', checkpoint_sha256='c' * 64)
        body, _ = reporter.render({'cause': {'checkpoint_bn_probe': probe}})
        self.assertIn('0.00103', body)
        self.assertIn('3/3 seed', body)
        self.assertIn('不接受 BN 校准为修复方案', body)
        self.assertNotIn('/private/checkpoint-sentinel', body)
        self.assertNotRegex(body, r'[0-9a-fA-F]{64}')

    def test_velocity_state_scope_and_counterevidence_are_preserved(self):
        evidence = SOURCE.parents[2] / 'research/staging/trajectory_shortening_20261024/velocity_state_observation_20261108.json'
        probe = json.loads(evidence.read_text())
        probe.update(leaf='/private/state-sentinel', checkpoint_sha256='d' * 64)
        body, _ = reporter.render({'cause': {'velocity_state_probe': probe}})
        for expected in ('720 窗', '0.1332', '1.7154', '0.1322', '构造的可辨识性检查', '特权信息'):
            self.assertIn(expected, body)
        self.assertNotIn('/private/state-sentinel', body)
        self.assertNotRegex(body, r'[0-9a-fA-F]{64}')
        probe['recordings'] = 1
        body, _ = reporter.render({'cause': {'velocity_state_probe': probe}})
        self.assertNotIn('初速度边界诊断', body)

    def test_radial_gain_counterexamples_keep_validation_scope_and_privacy(self):
        evidence = SOURCE.parents[2] / 'research/staging/trajectory_shortening_20261024/radial_loss_observation_20261108.json'
        probe = json.loads(evidence.read_text())
        probe.update(leaf='/private/radial-sentinel', source_sha256='e' * 64)
        body, _ = reporter.render({'cause': {'radial_loss_probe': probe}})
        for expected in ('3/3 seed', '2/3 seed', '验证仅一个 group', '窗口权重', '无测试集或 ATE/RTE 结论'):
            self.assertIn(expected, body)
        self.assertNotIn('/private/radial-sentinel', body)
        self.assertNotRegex(body, r'[0-9a-fA-F]{64}')
        probe['gain_fit_split'] = 'test'
        body, _ = reporter.render({'cause': {'radial_loss_probe': probe}})
        self.assertNotIn('径向损失诊断', body)


if __name__ == '__main__':
    unittest.main()
