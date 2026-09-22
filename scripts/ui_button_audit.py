#!/usr/bin/env python
"""Exercise every interactive control in the BlindGuard web UI.

This is an audit tool, not a strict pass/fail CI test. It records browser-side
state, backend state, console errors, and request failures so regressions in
button wiring can be reviewed together.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


def backend(url: str, path: str, method: str = "GET", payload=None):
    response = requests.request(
        method,
        f"{url.rstrip('/')}{path}",
        json=payload,
        timeout=10,
    )
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {"status": response.status_code, "body": body}


def button_state(page, selector: str):
    locator = page.locator(selector)
    return {
        "visible": locator.is_visible(),
        "enabled": locator.is_enabled(),
        "text": locator.inner_text().strip(),
    }


def wait_for_system(page, url: str, predicate, timeout=15):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = backend(url, "/api/system/status")["body"]
        if predicate(last):
            return last
        page.wait_for_timeout(300)
    return last


def completed_bot_count(page):
    return page.locator(".chat-msg.bot").evaluate_all(
        "(nodes) => nodes.filter("
        "(node) => !node.textContent.includes('思考中...')).length"
    )


def wait_for_next_bot_reply(page, completed_before, timeout=30000):
    page.wait_for_function(
        "(n) => [...document.querySelectorAll('.chat-msg.bot')].filter("
        "(node) => !node.textContent.includes('思考中...')).length > n",
        arg=completed_before,
        timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--video", default="uploads/video.mp4")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")

    report = {"url": args.url, "video": str(video), "checks": []}

    def check(name, passed, detail):
        report["checks"].append(
            {
                "name": name,
                "passed": None if passed is None else bool(passed),
                "detail": detail,
            }
        )

    # Keep the audit repeatable after an interrupted run. The API reset does not
    # replace any checks below; it only restores the documented initial state.
    for path in ("/api/video/stop", "/api/camera/stop", "/api/system/stop"):
        try:
            backend(args.url, path, method="POST")
        except requests.RequestException:
            pass

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=args.headless)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})

        # Deterministic microphone path: exercise the UI state machine without
        # depending on browser speech permissions or an external model download.
        context.add_init_script(
            """
            class MockSpeechRecognition {
              constructor() { this.lang=''; this.interimResults=false;
                this.maxAlternatives=1; this.onresult=null; this.onend=null;
                this.onerror=null; }
              start() { this._active=true; }
              stop() { this._active=false; if (this.onend) this.onend(); }
              abort() { this._active=false; if (this.onend) this.onend(); }
            }
            window.SpeechRecognition = MockSpeechRecognition;
            window.webkitSpeechRecognition = MockSpeechRecognition;
            """
        )

        console_errors = []
        page_errors = []
        failed_requests = []
        page = context.new_page()
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "requestfailed",
            lambda req: failed_requests.append(
                {"url": req.url, "failure": req.failure}
            ),
        )

        response = page.goto(args.url, wait_until="load")
        check(
            "首页加载",
            response is not None and response.ok,
            {"http_status": response.status if response else None},
        )
        page.wait_for_timeout(700)

        initial = {
            selector: button_state(page, selector)
            for selector in (
                "#btnStartSystem",
                "#btnStopSystem",
                "#btnStartCamera",
                "#btnStopCamera",
                "#btnStopVideo",
            )
        }
        check(
            "初始按钮状态",
            initial["#btnStartSystem"]["enabled"]
            and not initial["#btnStopSystem"]["enabled"]
            and not initial["#btnStartCamera"]["enabled"]
            and not initial["#btnStopCamera"]["enabled"]
            and not initial["#btnStopVideo"]["enabled"],
            initial,
        )

        page.locator("#btnContrast").click()
        contrast_on = page.locator("body").evaluate(
            "el => el.classList.contains('hc')"
        )
        contrast_styles = page.evaluate(
            """
            () => {
              const style = (selector) => getComputedStyle(
                document.querySelector(selector)
              );
              const button = document.getElementById('btnContrast');
              return {
                bodyFont: parseFloat(style('body').fontSize),
                messageFont: parseFloat(style('.message').fontSize),
                buttonFont: parseFloat(style('.btn').fontSize),
                titleFont: parseFloat(style('.logo h1').fontSize),
                bodyColor: style('body').color,
                bodyBackground: style('body').backgroundColor,
                pressed: button.getAttribute('aria-pressed'),
                label: button.getAttribute('aria-label'),
              };
            }
            """
        )
        page.reload(wait_until="load")
        contrast_persisted = page.locator("body").evaluate(
            "el => el.classList.contains('hc')"
        )
        check(
            "大字模式按钮",
            (
                contrast_on
                and contrast_persisted
                and contrast_styles["bodyFont"] >= 18
                and contrast_styles["messageFont"] >= 18
                and contrast_styles["buttonFont"] >= 18
                and contrast_styles["titleFont"] >= 26
                and contrast_styles["pressed"] == "true"
                and contrast_styles["label"] == "关闭大字模式"
            ),
            {
                "on": contrast_on,
                "persisted": contrast_persisted,
                **contrast_styles,
            },
        )
        page.locator("#btnContrast").click()

        page.locator("#btnOverlay").click()
        overlay_off = page.locator("#overlay").evaluate(
            "el => getComputedStyle(el).display === 'none'"
        )
        regex_label = page.locator("#btnOverlay").inner_text()
        page.reload(wait_until="load")
        overlay_persisted = page.locator("#overlay").evaluate(
            "el => getComputedStyle(el).display === 'none'"
        )
        check(
            "标签层按钮",
            overlay_off and overlay_persisted and "关" in regex_label,
            {
                "off": overlay_off,
                "persisted": overlay_persisted,
                "label": regex_label,
            },
        )
        page.locator("#btnOverlay").click()

        page.locator("#btnStartSystem").click()
        started = wait_for_system(page, args.url, lambda s: s.get("running"))
        page.wait_for_function(
            "() => !document.getElementById('btnStopSystem').disabled",
            timeout=5000,
        )
        check(
            "启动系统按钮",
            started.get("running")
            and page.locator("#btnStartSystem").is_disabled()
            and page.locator("#btnStopSystem").is_enabled(),
            {
                "backend": started,
                "button": button_state(page, "#btnStartSystem"),
                "stop_button": button_state(page, "#btnStopSystem"),
            },
        )

        page.reload(wait_until="load")
        page.wait_for_timeout(800)
        refreshed_ui = {
            "status": backend(args.url, "/api/system/status")["body"],
            "start_button": button_state(page, "#btnStartSystem"),
            "stop_button": button_state(page, "#btnStopSystem"),
        }
        check(
            "刷新后的运行状态恢复",
            refreshed_ui["status"].get("running")
            and refreshed_ui["start_button"]["enabled"] is False
            and refreshed_ui["stop_button"]["enabled"] is True,
            refreshed_ui,
        )
        if (
            refreshed_ui["status"].get("running")
            and refreshed_ui["start_button"]["enabled"]
        ):
            # Preserve the failed check above, then resynchronize the page so
            # the remaining controls can still be audited in this run.
            page.locator("#btnStartSystem").click()
            page.wait_for_function(
                "() => !document.getElementById('btnStopSystem').disabled",
                timeout=10000,
            )

        page.locator("#btnStartCamera").click()
        page.wait_for_function(
            "() => document.getElementById('messageArea').textContent"
            ".includes('摄像头')",
            timeout=10000,
        )
        camera_status = backend(args.url, "/api/system/status")["body"]
        camera_message = page.locator("#messageArea").inner_text()
        camera_started = bool(camera_status.get("camera_active"))
        check(
            "开启摄像头按钮",
            camera_started
            or "失败" in camera_message
            or "无法打开摄像头" in camera_message,
            {
                "started": camera_started,
                "message": camera_message,
                "button": button_state(page, "#btnStartCamera"),
            },
        )
        if camera_started:
            page.locator("#btnStopCamera").click()
            stopped_camera = wait_for_system(
                page, args.url, lambda s: not s.get("camera_active")
            )
            page.wait_for_timeout(1000)
            stopped_camera = backend(
                args.url, "/api/system/status"
            )["body"]
            check(
                "关闭摄像头按钮",
                not stopped_camera.get("camera_active")
                and stopped_camera.get("detection_count") == 0
                and stopped_camera.get("overall_risk") == "safe"
                and not stopped_camera.get("scene_type")
                and stopped_camera.get("fps") == 0
                and page.locator("#btnStopCamera").is_disabled(),
                {
                    "backend": stopped_camera,
                    "button": button_state(page, "#btnStopCamera"),
                },
            )
        else:
            check(
                "关闭摄像头按钮",
                page.locator("#btnStopCamera").is_disabled(),
                "本机没有可用摄像头，按钮按状态机保持禁用",
            )

        upload_area = page.locator(".upload-area")
        upload_area.focus()
        with page.expect_file_chooser() as chooser_info:
            page.keyboard.press("Enter")
        chooser_info.value.set_files(str(video))
        uploaded = wait_for_system(
            page,
            args.url,
            lambda s: s.get("video_active"),
            timeout=30,
        )
        controls_visible = page.locator("#videoControls").is_visible()
        check(
            "上传视频区域-键盘 Enter",
            uploaded.get("video_active") and controls_visible,
            {
                "backend": uploaded,
                "controls_visible": controls_visible,
                "key": "Enter",
            },
        )

        page.locator("#btnPause").click()
        paused = wait_for_system(
            page, args.url, lambda s: s.get("video_paused")
        )
        page.wait_for_function(
            "() => document.getElementById('btnPause').textContent.trim()"
            " === '播放'",
            timeout=5000,
        )
        check(
            "暂停/播放按钮-暂停",
            paused.get("video_paused")
            and page.locator("#btnPause").inner_text().strip() == "播放",
            {
                "backend": paused,
                "button": page.locator("#btnPause").inner_text().strip(),
            },
        )

        page.locator("#btnPause").click()
        resumed = wait_for_system(
            page, args.url, lambda s: not s.get("video_paused")
        )
        page.wait_for_function(
            "() => document.getElementById('btnPause').textContent.trim()"
            " === '暂停'",
            timeout=5000,
        )
        check(
            "暂停/播放按钮-继续",
            not resumed.get("video_paused")
            and page.locator("#btnPause").inner_text().strip() == "暂停",
            {
                "backend": resumed,
                "button": page.locator("#btnPause").inner_text().strip(),
            },
        )

        speed_results = {}
        for speed in (0.5, 1.0, 1.5, 2.0):
            speed_label = f"{speed:g}x"
            page.locator(".speed-btn", has_text=speed_label).click()
            state = wait_for_system(
                page,
                args.url,
                lambda s, wanted=speed: abs(
                    float(s.get("video_speed", 0)) - wanted
                )
                < 0.001,
            )
            page.wait_for_function(
                "(label) => document.querySelector('.speed-btn.active')"
                "?.textContent.trim() === label",
                arg=speed_label,
                timeout=5000,
            )
            speed_results[str(speed)] = {
                "backend": state.get("video_speed"),
                "active": page.locator(
                    ".speed-btn.active"
                ).inner_text().strip(),
            }
        check(
            "倍速按钮组",
            all(
                abs(float(item["backend"]) - float(speed)) < 0.001
                and item["active"] == f"{float(speed):g}x"
                for speed, item in speed_results.items()
            ),
            speed_results,
        )

        page.locator("#btnPause").click()
        wait_for_system(page, args.url, lambda s: s.get("video_paused"))
        before_seek = backend(args.url, "/api/system/status")["body"].get(
            "video_position", 0
        )
        progress_total = int(
            backend(args.url, "/api/system/status")["body"].get(
                "video_total", 0
            )
        )
        target_frame = int(progress_total * 0.7)
        frame_tolerance = max(3, int(progress_total * 0.01))
        bar = page.locator("#progressBar")
        box = bar.bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.7, box["y"] + 5)
        seeked = wait_for_system(
            page,
            args.url,
            lambda s: abs(
                int(s.get("video_position", 0)) - target_frame
            )
            <= frame_tolerance,
        )
        seek_error = abs(
            int(seeked.get("video_position", 0)) - target_frame
        )
        check(
            "进度条回跳",
            seek_error <= frame_tolerance,
            {
                "before": before_seek,
                "after": seeked.get("video_position"),
                "total": seeked.get("video_total"),
                "target": target_frame,
                "tolerance": frame_tolerance,
                "error": seek_error,
            },
        )

        keyboard_seek_results = []
        for key in (
            "Home",
            "End",
            "PageUp",
            "PageDown",
            "ArrowLeft",
            "ArrowRight",
        ):
            current = int(
                page.locator("#progressBar").get_attribute("aria-valuenow")
                or 0
            )
            page_step = max(1, round(progress_total * 0.1))
            line_step = max(1, round(progress_total * 0.01))
            if key == "Home":
                target = 0
            elif key == "End":
                target = progress_total - 1
            elif key == "PageUp":
                target = current - page_step
            elif key == "PageDown":
                target = current + page_step
            elif key == "ArrowLeft":
                target = current - line_step
            else:
                target = current + line_step
            target = min(progress_total - 1, max(0, target))

            with page.expect_response(
                lambda response: response.url.endswith("/api/video/seek")
            ) as seek_response:
                page.locator("#progressBar").press(key)
            response_body = seek_response.value.json()
            keyboard_state = wait_for_system(
                page,
                args.url,
                lambda s, wanted=target: abs(
                    int(s.get("video_position", 0)) - wanted
                )
                <= frame_tolerance,
            )
            keyboard_seek_results.append(
                {
                    "key": key,
                    "before": current,
                    "target": target,
                    "after": keyboard_state.get("video_position"),
                    "api_success": response_body.get("success"),
                    "error": abs(
                        int(keyboard_state.get("video_position", 0)) - target
                    ),
                }
            )
        check(
            "进度条键盘回跳",
            all(
                item["api_success"]
                and item["error"] <= frame_tolerance
                for item in keyboard_seek_results
            ),
            {
                "tolerance": frame_tolerance,
                "results": keyboard_seek_results,
            },
        )
        page.locator("#btnPause").click()
        wait_for_system(page, args.url, lambda s: not s.get("video_paused"))

        try:
            page.wait_for_selector(
                ".timeline-item.seekable",
                timeout=8000,
            )
        except PlaywrightTimeoutError:
            pass
        timeline_items = page.locator(".timeline-item.seekable")
        timeline_count = timeline_items.count()
        if timeline_count:
            timeline_status = backend(args.url, "/api/system/status")["body"]
            timeline_events = [
                event
                for event in timeline_status.get("events", [])
                if event.get("frame")
            ]
            if timeline_events:
                target_event = timeline_events[0]
                target_event_frame = int(target_event["frame"])
                timeline_total = int(
                    timeline_status.get("video_total", 0)
                )
                timeline_tolerance = max(
                    3, int(timeline_total * 0.01)
                )
                event_item = page.locator(
                    ".timeline-item.seekable",
                    has_text=target_event.get("text", ""),
                ).first
                event_item.focus()
                with page.expect_response(
                    lambda response: response.url.endswith("/api/video/seek")
                ) as enter_response:
                    page.keyboard.press("Enter")
                enter_body = enter_response.value.json()
                page.wait_for_function(
                    "([target, tolerance]) => Math.abs("
                    "Number(document.getElementById('progressBar')"
                    ".getAttribute('aria-valuenow')) - target) <= tolerance",
                    arg=[target_event_frame, timeline_tolerance],
                    timeout=5000,
                )
                timeline_seeked = wait_for_system(
                    page,
                    args.url,
                    lambda s: abs(
                        int(s.get("video_position", 0))
                        - target_event_frame
                    )
                    <= timeline_tolerance,
                )
                timeline_error = abs(
                    int(timeline_seeked.get("video_position", 0))
                    - target_event_frame
                )
                check(
                    "时间线回跳-键盘 Enter/Space",
                    timeline_error <= timeline_tolerance
                    and enter_body.get("success"),
                    {
                        "key": ["Enter", "Space"],
                        "target": target_event_frame,
                        "after": timeline_seeked.get("video_position"),
                        "tolerance": timeline_tolerance,
                        "error": timeline_error,
                        "enter_api_success": enter_body.get("success"),
                    },
                )
                with page.expect_response(
                    lambda response: response.url.endswith("/api/video/seek")
                ) as space_response:
                    page.keyboard.press("Space")
                space_body = space_response.value.json()
                check(
                    "时间线回跳-键盘 Space 独立触发",
                    bool(space_body.get("success")),
                    {"space_api_success": space_body.get("success")},
                )
            else:
                check(
                    "时间线回跳",
                    None,
                    "时间线存在，但没有携带帧号的主动播报事件",
                )
        else:
            check(
                "时间线回跳",
                None,
                "本次短测没有产生可回跳的主动播报事件",
            )

        page.locator("#btnStopVideo").click()
        stopped_video = wait_for_system(
            page, args.url, lambda s: not s.get("video_active")
        )
        page.wait_for_timeout(1000)
        stopped_video = backend(args.url, "/api/system/status")["body"]
        page.wait_for_function(
            "() => getComputedStyle(document.getElementById('videoControls'))"
            ".display === 'none'",
            timeout=5000,
        )
        check(
            "停止视频按钮",
            not stopped_video.get("video_active")
            and not page.locator("#videoControls").is_visible()
            and stopped_video.get("detection_count") == 0
            and stopped_video.get("overall_risk") == "safe"
            and not stopped_video.get("scene_type")
            and stopped_video.get("fps") == 0,
            {
                "backend": stopped_video,
                "controls_visible": page.locator("#videoControls").is_visible(),
                "stale_detection_count": stopped_video.get(
                    "detection_count"
                ),
            },
        )

        # 停止前保持 2x；新视频必须恢复为后端实际使用的 1x。
        upload_area.focus()
        with page.expect_file_chooser() as chooser_info:
            page.keyboard.press("Space")
        chooser_info.value.set_files(str(video))
        reuploaded = wait_for_system(
            page, args.url, lambda s: s.get("video_active")
        )
        page.wait_for_function(
            "() => document.querySelector('.speed-btn.active')"
            "?.textContent.trim() === '1x'",
            timeout=5000,
        )
        check(
            "重新上传后倍速归位",
            abs(float(reuploaded.get("video_speed", 0)) - 1.0) < 0.001
            and page.locator(".speed-btn.active").inner_text().strip() == "1x",
            {
                "backend": reuploaded,
                "active": page.locator(".speed-btn.active").inner_text().strip(),
            },
        )
        page.locator("#btnStopVideo").click()
        wait_for_system(
            page, args.url, lambda s: not s.get("video_active")
        )
        page.wait_for_timeout(500)

        questions = (
            "前面有什么",
            "现在能过马路吗",
            "周围有什么危险",
            "读一下路牌上的字",
        )
        quick_results = []
        for question in questions:
            before = completed_bot_count(page)
            page.locator(".chip", has_text=question).click()
            wait_for_next_bot_reply(page, before)
            quick_results.append(
                {
                    "question": question,
                    "last_reply": page.locator(".chat-msg.bot").last.inner_text(),
                }
            )
        check(
            "快捷提问按钮组",
            len(quick_results) == 4
            and all(item["last_reply"].strip() for item in quick_results),
            quick_results,
        )

        agent_status = backend(args.url, "/api/agent/status")["body"]
        if not agent_status.get("enabled"):
            expected_badge = "未启用"
        elif agent_status.get("llm_active"):
            expected_badge = agent_status.get("model", "")
            if agent_status.get("vision_active"):
                expected_badge += " · 视觉"
        else:
            expected_badge = "本地回退"
        page.wait_for_function(
            "(expected) => document.getElementById('agentBadge')"
            ".textContent.trim() === expected",
            arg=expected_badge,
            timeout=6000,
        )
        check(
            "智能体状态徽标",
            page.locator("#agentBadge").inner_text().strip()
            == expected_badge,
            {
                "backend": agent_status,
                "expected": expected_badge,
                "actual": page.locator("#agentBadge").inner_text().strip(),
            },
        )

        page.locator("#chatInput").fill("前面有什么")
        completed_before = completed_bot_count(page)
        page.locator("#btnSend").click()
        wait_for_next_bot_reply(page, completed_before)
        check(
            "发送按钮",
            page.locator(".chat-msg.user").last.inner_text().endswith(
                "前面有什么"
            ),
            page.locator(".chat-msg").last.inner_text(),
        )

        completed_before = completed_bot_count(page)
        page.locator("#chatInput").fill("前面有什么")
        page.locator("#btnSend").click()
        page.locator("#chatInput").fill("现在能过马路吗")
        page.locator("#btnSend").click()
        page.wait_for_function(
            "(n) => [...document.querySelectorAll('.chat-msg.bot')].filter("
            "(node) => !node.textContent.includes('思考中...')).length >= n",
            arg=completed_before + 2,
            timeout=30000,
        )
        sent_users = page.locator(".chat-msg.user").evaluate_all(
            "(nodes) => nodes.slice(-2).map((node) => node.textContent)"
        )
        check(
            "连续发送顺序",
            sent_users[-2].endswith("前面有什么")
            and sent_users[-1].endswith("现在能过马路吗")
            and "思考中..." not in page.locator("#chatBox").inner_text(),
            sent_users,
        )

        page.locator("#btnMic").click()
        mic_listening = page.locator("#btnMic").evaluate(
            "el => el.classList.contains('listening')"
        )
        page.locator("#btnMic").click()
        mic_stopped = not page.locator("#btnMic").evaluate(
            "el => el.classList.contains('listening')"
        )
        check(
            "语音输入按钮",
            mic_listening and mic_stopped,
            {"listening": mic_listening, "stopped": mic_stopped},
        )

        page.locator("#btnStopSystem").click()
        stopped = wait_for_system(
            page, args.url, lambda s: not s.get("running")
        )
        page.wait_for_function(
            "() => !document.getElementById('btnStartSystem').disabled"
            " && document.getElementById('btnStopSystem').disabled"
            " && getComputedStyle(document.getElementById('videoControls'))"
            ".display === 'none'",
            timeout=5000,
        )
        stopped_ui = {
            "fps": page.locator("#fpsValue").inner_text(),
            "objects": page.locator("#objectCount").inner_text(),
            "risk": page.locator("#riskLevel").inner_text(),
            "scene": page.locator("#sceneType").inner_text(),
            "detections": page.locator("#detectionList").inner_text(),
            "timeline": page.locator("#timeline").inner_text(),
            "video_feed_visibility": page.locator("#videoFeed").evaluate(
                "el => getComputedStyle(el).visibility"
            ),
            "overlay_ink": page.locator("#overlay").evaluate(
                """(canvas) => {
                    const ctx = canvas.getContext('2d');
                    const data = ctx.getImageData(
                        0, 0, canvas.width, canvas.height
                    ).data;
                    let ink = 0;
                    for (let i = 3; i < data.length; i += 4) {
                        if (data[i] !== 0) ink += 1;
                    }
                    return ink;
                }"""
            ),
        }
        check(
            "停止系统按钮",
            not stopped.get("running")
            and page.locator("#btnStartSystem").is_enabled()
            and page.locator("#btnStopSystem").is_disabled()
            and not page.locator("#videoControls").is_visible()
            and stopped_ui["fps"] == "0"
            and stopped_ui["objects"] == "0"
            and stopped_ui["risk"] == "安全"
            and stopped_ui["scene"] == "-"
            and "暂无检测结果" in stopped_ui["detections"]
            and stopped_ui["video_feed_visibility"] == "hidden"
            and stopped_ui["overlay_ink"] == 0,
            {
                "backend": stopped,
                "frontend": stopped_ui,
                "start_button": button_state(page, "#btnStartSystem"),
                "stop_button": button_state(page, "#btnStopSystem"),
            },
        )

        page.screenshot(
            path="evaluation/ui_button_audit.png", full_page=True
        )
        report["console_errors"] = console_errors
        report["page_errors"] = page_errors
        report["failed_requests"] = failed_requests
        report["screenshot"] = str(
            Path("evaluation/ui_button_audit.png").resolve()
        )
        browser.close()

    output = Path("evaluation/ui_button_audit.json")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    passed = sum(item["passed"] is True for item in report["checks"])
    failed = sum(item["passed"] is False for item in report["checks"])
    skipped = sum(item["passed"] is None for item in report["checks"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"\nSummary: {passed} passed, {failed} failed, "
        f"{skipped} not exercised"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
