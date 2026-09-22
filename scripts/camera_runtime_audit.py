#!/usr/bin/env python
"""Exercise the live camera path through the running BlindGuard API.

The audit measures status updates, frame delivery, inference progress, and
cleanup. If the host has no usable camera, the report records an environment
skip instead of claiming success.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def request_json(base_url, path, method="GET", payload=None, timeout=15):
    response = requests.request(
        method,
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=timeout,
    )
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def add_check(checks, name, passed, detail):
    checks.append({
        "name": name,
        "passed": None if passed is None else bool(passed),
        "detail": detail,
    })


def wait_status(base_url, predicate, timeout=15):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        _, last = request_json(base_url, "/api/system/status")
        if predicate(last):
            return last
        time.sleep(0.25)
    return last


def capture_first_jpeg(base_url, timeout=8):
    """Read one JPEG from the MJPEG endpoint without relying on a browser."""
    url = f"{base_url.rstrip('/')}/video_feed"
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                return None, f"HTTP {response.status_code}"
            content_type = response.headers.get("Content-Type", "")
            buffer = bytearray()
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                buffer.extend(chunk)
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", max(0, start + 2))
                if start >= 0 and end > start:
                    return bytes(buffer[start:end + 2]), content_type
                if len(buffer) > 4 * 1024 * 1024:
                    return None, "JPEG frame exceeded 4 MiB"
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return None, "stream ended before a JPEG frame arrived"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument(
        "--output",
        default="evaluation/camera_runtime_audit.json",
    )
    parser.add_argument(
        "--frame-output",
        default="evaluation/camera_runtime_audit_frame.jpg",
    )
    args = parser.parse_args()

    checks = []
    samples = []
    frame_path = Path(args.frame_output)

    try:
        _, initial = request_json(args.url, "/api/system/status")
        if initial.get("running"):
            request_json(args.url, "/api/system/stop", method="POST")
            wait_status(args.url, lambda status: not status.get("running"))

        status_code, start = request_json(
            args.url, "/api/system/start", method="POST"
        )
        add_check(
            checks,
            "start_system_api",
            status_code == 200 and start.get("success"),
            {"status_code": status_code, "body": start},
        )

        status_code, camera = request_json(
            args.url, "/api/camera/start", method="POST"
        )
        camera_started = status_code == 200 and camera.get("success")
        add_check(
            checks,
            "start_camera_api",
            camera_started,
            {
                "status_code": status_code,
                "body": camera,
                "environment_result": (
                    "camera unavailable"
                    if not camera_started
                    else "camera opened"
                ),
            },
        )

        if not camera_started:
            add_check(
                checks,
                "camera_inference",
                None,
                "No camera is available to this process/user.",
            )
            add_check(
                checks,
                "camera_frame_delivery",
                None,
                "Frame delivery cannot be checked without camera access.",
            )
            add_check(
                checks,
                "camera_cleanup",
                True,
                "Camera was not acquired; no cleanup was required.",
            )
        else:
            deadline = time.time() + args.settle_seconds
            while time.time() < deadline:
                _, status = request_json(args.url, "/api/system/status")
                samples.append({
                    "t": round(time.time(), 3),
                    "fps": status.get("fps"),
                    "detection_count": status.get("detection_count"),
                    "overall_risk": status.get("overall_risk"),
                })
                time.sleep(1.0)

            final_status = samples[-1] if samples else {}
            positive_fps = any(float(sample.get("fps") or 0) > 0 for sample in samples)
            add_check(
                checks,
                "camera_inference",
                positive_fps,
                {
                    "samples": samples,
                    "positive_fps": positive_fps,
                },
            )

            jpeg, detail = capture_first_jpeg(args.url)
            if jpeg:
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(jpeg)
            add_check(
                checks,
                "camera_frame_delivery",
                bool(jpeg and jpeg.startswith(b"\xff\xd8")),
                {
                    "detail": detail,
                    "bytes": len(jpeg) if jpeg else 0,
                    "frame_output": str(frame_path.resolve()) if jpeg else "",
                    "latest_status": final_status,
                },
            )

            status_code, stop_camera = request_json(
                args.url, "/api/camera/stop", method="POST"
            )
            stopped = wait_status(
                args.url,
                lambda status: not status.get("camera_active"),
            )
            add_check(
                checks,
                "camera_cleanup",
                (
                    status_code == 200
                    and stop_camera.get("success")
                    and not stopped.get("camera_active")
                    and stopped.get("detection_count") == 0
                    and stopped.get("fps") == 0
                ),
                {
                    "status_code": status_code,
                    "response": stop_camera,
                    "status": stopped,
                },
            )
    finally:
        try:
            request_json(args.url, "/api/system/stop", method="POST")
        except requests.RequestException:
            pass

    report = {
        "url": args.url,
        "checks": checks,
        "frame_output": (
            str(frame_path.resolve()) if frame_path.exists() else ""
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = sum(item["passed"] is True for item in checks)
    failed = sum(item["passed"] is False for item in checks)
    skipped = sum(item["passed"] is None for item in checks)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"\nSummary: {passed} passed, {failed} failed, "
        f"{skipped} environment-skipped"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
