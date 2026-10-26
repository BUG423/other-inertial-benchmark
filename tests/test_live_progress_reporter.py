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


if __name__ == '__main__':
    unittest.main()
