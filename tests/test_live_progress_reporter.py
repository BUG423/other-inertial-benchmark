"""Reporter privacy and command quoting without a network connection."""

import ast
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'tools/ops_v1/live_progress_reporter.py'
SPEC = importlib.util.spec_from_file_location('live_progress_reporter', SOURCE)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class ReporterPrivacyTests(unittest.TestCase):
    def test_public_report_omits_private_runtime_fields(self):
        body, _ = reporter.render({
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
