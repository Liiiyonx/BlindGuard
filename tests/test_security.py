# -*- coding: utf-8 -*-
"""Tests for server bind validation and token authentication."""

import unittest
from unittest.mock import patch

from core.security import (
    is_loopback_host,
    is_protected_path,
    extract_token,
    token_is_valid,
    validate_server_binding,
)


class SecurityTests(unittest.TestCase):
    def test_loopback_hosts_are_detected(self):
        for host in ('127.0.0.1', '127.8.9.10', 'localhost', '::1', '[::1]'):
            with self.subTest(host=host):
                self.assertTrue(is_loopback_host(host))

    def test_non_loopback_hosts_are_rejected(self):
        for host in ('0.0.0.0', '192.168.1.20', '10.0.0.5', 'example.local', ''):
            with self.subTest(host=host):
                self.assertFalse(is_loopback_host(host))

    def test_non_loopback_bind_requires_token(self):
        with self.assertRaises(RuntimeError):
            validate_server_binding('0.0.0.0', '')
        with self.assertRaises(RuntimeError):
            validate_server_binding('0.0.0.0', '   ')

        validate_server_binding('127.0.0.1', '')
        validate_server_binding('0.0.0.0', 'long-enough-token')

    def test_extract_token_priority_and_formats(self):
        headers = {
            'X-BlindGuard-Token': 'header-token',
            'Authorization': 'Bearer bearer-token',
        }
        self.assertEqual(
            extract_token(headers, {'token': 'query-token'}),
            'header-token',
        )
        self.assertEqual(
            extract_token({'Authorization': 'bEaReR bearer-token'}, {}),
            'bearer-token',
        )
        self.assertEqual(extract_token({}, {'token': 'query-token'}), 'query-token')
        self.assertEqual(extract_token({}, {}), '')

    def test_token_validation_uses_constant_time_comparison(self):
        self.assertTrue(token_is_valid('anything', ''))
        self.assertFalse(token_is_valid('', 'expected'))

        with patch('core.security.secrets.compare_digest', return_value=True) as compare:
            self.assertTrue(token_is_valid('provided', 'expected'))
            compare.assert_called_once_with('provided', 'expected')

    def test_only_sensitive_paths_are_protected(self):
        self.assertTrue(is_protected_path('/'))
        self.assertTrue(is_protected_path('/video_feed'))
        self.assertTrue(is_protected_path('/api/detections'))
        self.assertFalse(is_protected_path('/sw.js'))
        self.assertFalse(is_protected_path('/icon_192.png'))
        self.assertFalse(is_protected_path('/favicon.ico'))


if __name__ == '__main__':
    unittest.main()
