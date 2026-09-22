# -*- coding: utf-8 -*-
"""Static regression checks for authenticated frontend requests."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TemplateTests(unittest.TestCase):
    def test_api_requests_use_token_header_helper(self):
        html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('<link rel="icon"', html)
        self.assertIn('function apiFetch(', html)
        self.assertIn('X-BlindGuard-Token', html)
        self.assertIn('AUTH_TOKEN', html)
        self.assertIn('conf_stable', html)
        self.assertIn('conf_smooth', html)

    def test_frontend_does_not_inject_dynamic_html(self):
        html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertNotIn('innerHTML', html)

    def test_frontend_states_safety_boundary_without_approval_copy(self):
        html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('环境辅助提示，不构成通行许可', html)
        self.assertIn('不能批准通行', html)
        for phrase in ('可以正常通行', '可以安全通行', '放心走'):
            self.assertNotIn(phrase, html)

    def test_service_worker_never_caches_protected_or_token_urls(self):
        source = (ROOT / 'static' / 'sw.js').read_text(encoding='utf-8')
        self.assertIn("url.pathname === '/'", source)
        self.assertIn("url.pathname === '/video_feed'", source)
        self.assertIn("url.pathname.startsWith('/api/')", source)
        self.assertIn("url.searchParams.has('token')", source)

    def test_frontend_recovers_backend_state_and_serializes_chat(self):
        html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('async function syncSystemState()', html)
        self.assertIn('await syncSystemState()', html)
        self.assertIn('let chatRequestQueue = Promise.resolve()', html)
        self.assertIn('replaceChatMessage(pending', html)
        self.assertNotIn("box.lastChild.remove()", html)

    def test_frontend_exposes_keyboard_and_touch_accessible_controls(self):
        html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('id="progressBar" role="slider" tabindex="0"', html)
        self.assertIn('onkeydown="seekVideoByKeyboard(event)"', html)
        self.assertIn('class="upload-area" role="button" tabindex="0"', html)
        self.assertIn("event.key === 'Enter' || event.key === ' '", html)
        self.assertGreaterEqual(html.count('min-height: 44px;'), 3)

    def test_high_contrast_mode_scales_text_and_keeps_semantic_colors(self):
        html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('body.hc', html)
        self.assertIn('font-size: 18px;', html)
        self.assertIn('body.hc .status-value', html)
        self.assertIn('body.hc .message.warning', html)
        self.assertIn('color: #30d158 !important;', html)
        self.assertIn('color: #ff453a !important;', html)
        self.assertIn("localStorage.setItem('bg_hc'", html)
        self.assertIn("button.setAttribute('aria-label'", html)

    def test_stopping_video_clears_backend_analysis_state(self):
        source = (ROOT / 'app.py').read_text(encoding='utf-8')
        method = source.split('def _video_stop(self):', 1)[1].split(
            'def _stop_video(self):', 1
        )[0]
        self.assertIn('_reset_analysis_state()', method)


if __name__ == '__main__':
    unittest.main()
