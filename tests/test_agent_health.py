# -*- coding: utf-8 -*-
"""Deterministic tests for online-model health and fallback reporting."""

import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from core.agent import BlindGuardAgent


class AgentHealthTests(unittest.TestCase):
    @staticmethod
    def make_agent():
        return BlindGuardAgent({
            'enabled': True,
            'llm': {
                'base_url': 'https://example.invalid/v1',
                'api_key': 'test-key',
                'model': 'test-model',
                'timeout': 0.01,
            },
            'announce': {
                'min_level': 'medium',
                'min_confidence': 0.45,
            },
        })

    def test_http_failure_switches_status_to_local_fallback(self):
        agent = self.make_agent()

        def raise_payment_required(*args, **kwargs):
            body = io.BytesIO(json.dumps({
                'error': {'message': 'Insufficient Balance'},
            }).encode('utf-8'))
            raise HTTPError(
                'https://example.invalid/v1/chat/completions',
                402,
                'Payment Required',
                {},
                body,
            )

        with patch(
            'core.agent.urllib.request.urlopen',
            side_effect=raise_payment_required,
        ):
            reply = agent.chat('周围有什么？', [])

        self.assertTrue(reply)
        self.assertFalse(agent.llm_available())
        status = agent.status()
        self.assertTrue(status['llm_configured'])
        self.assertFalse(status['llm_active'])
        self.assertTrue(status['fallback_active'])
        self.assertEqual(status['model'], '本地回退')
        self.assertIn('HTTP 402', status['last_error'])

    def test_successful_call_clears_degraded_status(self):
        agent = self.make_agent()
        agent._register_failure('HTTP 503: unavailable')
        self.assertFalse(agent.llm_available())

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            @staticmethod
            def read():
                return json.dumps({
                    'choices': [{'message': {'content': '模型已恢复'}}],
                }).encode('utf-8')

        with patch('core.agent.urllib.request.urlopen', return_value=Response()):
            message = agent._call_llm([{'role': 'user', 'content': 'test'}])

        self.assertEqual(message['content'], '模型已恢复')
        self.assertTrue(agent.llm_available())
        self.assertEqual(agent.status()['last_error'], '')


if __name__ == '__main__':
    unittest.main()
