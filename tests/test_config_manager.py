# -*- coding: utf-8 -*-
"""Tests for legacy YAML structure conversion."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.config_manager import ConfigManager


class ConfigManagerTests(unittest.TestCase):
    def _load(self, yaml_text):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / 'config.yaml'
        path.write_text(yaml_text, encoding='utf-8')
        with patch.object(ConfigManager, '_load_dotenv', return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                return ConfigManager(str(path))

    def test_defaults_use_current_dual_model_configuration(self):
        config = self._load('{}\n')
        self.assertEqual(config.get('model.path'), 'best_s.pt')
        self.assertEqual(config.get('model.aux_model_path'), 'best.pt')
        self.assertEqual(config.get('model.img_size'), 960)
        self.assertTrue(config.get('model.portrait_crop'))

    def test_legacy_detector_config_is_converted_without_losing_extensions(self):
        config = self._load(
            """
camera:
  index: 2
detector:
  model_path: models/main.pt
  confidence: 0.61
  iou_threshold: 0.42
  device: cpu
  img_size: 888
  focus_classes: [car, person]
  aux_model_path: models/aux.pt
  aux_classes: [manhole_cover]
  portrait_crop: true
risk_engine:
  area_thresholds:
    very_close: 0.18
  high_risk_classes: [car]
announcer:
  rate: 170
  volume: 0.8
""".strip()
        )

        self.assertEqual(config.get('model.path'), 'models/main.pt')
        self.assertEqual(config.get('model.confidence_threshold'), 0.61)
        self.assertEqual(config.get('model.iou_threshold'), 0.42)
        self.assertEqual(config.get('model.device'), 'cpu')
        self.assertEqual(config.get('model.img_size'), 888)
        self.assertEqual(config.get('model.focus_classes'), ['car', 'person'])
        self.assertEqual(config.get('model.aux_model_path'), 'models/aux.pt')
        self.assertEqual(config.get('model.aux_classes'), ['manhole_cover'])
        self.assertTrue(config.get('model.portrait_crop'))
        self.assertEqual(config.get('camera.device_id'), 2)
        self.assertEqual(config.get('risk.thresholds.very_close'), 0.18)
        self.assertEqual(config.get('voice.rate'), 170)
        self.assertEqual(config.get('voice.volume'), 0.8)


if __name__ == '__main__':
    unittest.main()
