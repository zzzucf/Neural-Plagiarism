import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiment_cli import parse_args
from experiment_report import save_result

ROOT = Path(__file__).resolve().parents[1]


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.inputs = self.folder / 'inputs'
        self.inputs.mkdir()
        (self.inputs / 'b.PNG').touch()
        (self.inputs / 'a.jpg').touch()
        (self.inputs / 'notes.txt').touch()
        (self.inputs / 'directory.jpg').mkdir()
        self.base = ['--target_folder', str(self.inputs), '--output_folder', str(self.folder / 'out')]

    def invalid(self, flags):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            parse_args(self.base + flags)
        self.assertEqual(exc.exception.code, 2)

    def test_original_example_single_bound_broadcasts(self):
        args, paths = parse_args(self.base + ['--start_step', '15', '--k', '25', '45', '--eps', '10'])
        self.assertEqual(args.eps, [10, 10])
        self.assertEqual([Path(p).name for p in paths], ['a.jpg', 'b.PNG'])

    def test_multiple_bounds_are_preserved(self):
        args, _ = parse_args(self.base + ['--eps', '10', '15'])
        self.assertEqual(args.eps, [10, 15])
        self.invalid(['--eps', '1', '2', '3'])

    def test_steps_cannot_skip_optimization_or_index_wrong_anchor(self):
        for flags in [['--k', '0'], ['--k', '50'], ['--start_step', '30', '--k', '25'],
                      ['--shortcut_step', '20'], ['--k', '25', '25'], ['--iters', '0']]:
            with self.subTest(flags=flags):
                self.invalid(flags)

    def test_bad_numerics_and_empty_ranges_fail_early(self):
        for flags in [['--image_length', '513'], ['--eps', 'nan'], ['--lr', 'inf'],
                      ['--guidance_scale', '1'], ['--start', '3'], ['--num_images', '0']]:
            with self.subTest(flags=flags):
                self.invalid(flags)

    def test_existing_results_are_not_overwritten(self):
        out = self.folder / 'out'
        out.mkdir()
        marker = out / 'previous-result.png'
        marker.write_bytes(b'previous experiment')
        self.invalid([])
        self.assertEqual(marker.read_bytes(), b'previous experiment')

    def test_help_and_dry_run_work_without_site_packages(self):
        help_run = subprocess.run([sys.executable, '-S', str(ROOT / 'run_attack.py'), '--help'],
                                  capture_output=True, text=True)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        run = subprocess.run([sys.executable, '-S', str(ROOT / 'run_attack.py'), *self.base,
                              '--k', '25', '45', '--eps', '10', '--dry-run'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        config = json.loads(run.stdout)
        self.assertEqual(config['arguments']['eps'], [10, 10])
        self.assertFalse((self.folder / 'out').exists())

    def test_report_preserves_results_and_escapes_filenames(self):
        rows = [dict(source='<test>.png', seed=0, input='image_input_0000.png',
                     reconstruction='reversed/image_0000_00.png', output='image_attack_0000_00.png')]
        save_result(self.folder, rows)
        rows.append(dict(rows[0], seed=10, reconstruction='reversed/image_0000_01.png',
                         output='image_attack_0000_01.png'))
        save_result(self.folder, rows)
        page = (self.folder / 'index.html').read_text(encoding='utf-8')
        self.assertIn('&lt;test&gt;.png', page)
        self.assertIn('image_0000_00.png', page)
        self.assertIn('image_0000_01.png', page)
        self.assertEqual(len(json.loads((self.folder / 'results.json').read_text())), 2)
        self.assertIn('does not compute watermark detection', page)


if __name__ == '__main__':
    unittest.main()
