# -*- coding: utf-8 -*-
"""Tests for POV dataset consent-scope validation."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_pov_dataset import validate


def make_record(split, consent_scope, sample_id='sample-1'):
    return {
        'sample_id': sample_id,
        'media_path': f'media/{sample_id}.jpg',
        'consent_id': 'consent-1',
        'consent_scope': consent_scope,
        'privacy_status': 'redacted',
        'viewpoint': 'first_person',
        'split': split,
        'scene_group': f'scene-{sample_id}',
    }


class PovAuditTests(unittest.TestCase):
    def audit_records(self, records):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            media_dir = root / 'media'
            media_dir.mkdir()
            for record in records:
                (root / record['media_path']).write_bytes(b'image')
            manifest_path = root / 'pov_manifest.json'
            manifest_path.write_text(
                json.dumps({'schema_version': 1, 'records': records}),
                encoding='utf-8',
            )
            audit, _ = validate(root, manifest_path)
            return audit

    def test_test_split_requires_evaluation_scope(self):
        audit = self.audit_records([
            make_record('test', ['training']),
        ])

        self.assertIn(
            'sample-1: test 数据缺少 evaluation 授权',
            audit.errors,
        )

    def test_test_split_accepts_evaluation_scope(self):
        audit = self.audit_records([
            make_record('test', ['research', 'evaluation']),
        ])

        self.assertFalse(
            any('数据缺少' in error for error in audit.errors),
            audit.errors,
        )

    def test_training_split_accepts_training_scope(self):
        audit = self.audit_records([
            make_record('train', ['research', 'training']),
        ])

        self.assertFalse(
            any('数据缺少' in error for error in audit.errors),
            audit.errors,
        )


if __name__ == '__main__':
    unittest.main()
