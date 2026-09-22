# -*- coding: utf-8 -*-
"""Tests for config-relative model path validation."""

import tempfile
import unittest
from pathlib import Path

from core.model_validation import (
    ModelValidationError,
    configured_model_files,
    missing_model_files,
    resolve_config_relative_path,
    validate_model_files,
)


class FakeConfig:
    def __init__(self, config_path, values=None):
        self.config_path = Path(config_path)
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


class ModelValidationTests(unittest.TestCase):
    def test_relative_paths_resolve_from_config_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / 'config' / 'config.yaml'
            resolved = resolve_config_relative_path(
                'models/main.pt', str(config_path))
            self.assertEqual(resolved, (root / 'config' / 'models' / 'main.pt').resolve())

    def test_absolute_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = (Path(temp_dir) / 'main.pt').resolve()
            resolved = resolve_config_relative_path(
                str(model_path), str(Path(temp_dir) / 'config.yaml'))
            self.assertEqual(resolved, model_path)

    def test_missing_files_reports_main_and_aux_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = FakeConfig(
                root / 'config.yaml',
                {
                    'model.path': 'models/main.pt',
                    'model.aux_model_path': 'models/aux.pt',
                },
            )

            entries = configured_model_files(config)
            self.assertEqual([role for role, _ in entries], ['主模型', '辅助模型'])
            self.assertEqual(len(missing_model_files(config)), 2)
            with self.assertRaises(ModelValidationError) as raised:
                validate_model_files(config)
            self.assertIn('主模型', str(raised.exception))
            self.assertIn('辅助模型', str(raised.exception))

    def test_existing_files_pass_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main_model = root / 'main.pt'
            aux_model = root / 'aux.pt'
            main_model.write_bytes(b'main')
            aux_model.write_bytes(b'aux')
            config = FakeConfig(
                root / 'config.yaml',
                {
                    'model.path': 'main.pt',
                    'model.aux_model_path': 'aux.pt',
                },
            )

            validate_model_files(config)
            self.assertEqual(missing_model_files(config), [])


if __name__ == '__main__':
    unittest.main()
