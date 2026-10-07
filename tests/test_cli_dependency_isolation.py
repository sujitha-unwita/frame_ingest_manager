"""Ensure doctor remains usable before optional dependencies are installed."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class DoctorDependencyTests(unittest.TestCase):
    """Run both entry points with server, panel, and serial imports blocked."""

    def test_doctor_reports_missing_dependencies(self) -> None:
        """Mock diagnostics succeed; real diagnostics fail cleanly, not on import."""
        workspace = Path(__file__).resolve().parents[2]
        script = textwrap.dedent('''
            import importlib.abc
            import runpy
            import sys

            class MissingDependencies(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {
                        'fastapi', 'uvicorn', 'websockets', 'serial',
                        'self_test', 'app_orchestrator', 'dashboard_service',
                    }:
                        raise ModuleNotFoundError(fullname)

            sys.meta_path.insert(0, MissingDependencies())
            sys.argv = ['main.py', 'doctor', sys.argv[1]]
            runpy.run_path('main.py', run_name='__main__')
        ''')
        for directory in (workspace, workspace / 'integrated_code'):
            for mode, expected_code in (('--mock', 0), ('--real', 1)):
                with self.subTest(directory=directory, mode=mode):
                    result = subprocess.run(
                        [sys.executable, '-c', script, mode],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, expected_code,
                                     result.stdout + result.stderr)
                    for dependency in ('fastapi', 'uvicorn', 'websockets', 'serial'):
                        self.assertRegex(result.stdout,
                                         dependency + r'\s+MISSING')
                    self.assertNotIn('Traceback', result.stderr)
