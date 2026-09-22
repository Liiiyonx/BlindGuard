#!/usr/bin/env python
"""Audit responsive layout and keyboard/touch accessibility.

This script is intentionally read-only with respect to application state. It
checks representative desktop and mobile widths for overflow, overlap, clipped
text, touch target size, accessible naming, focus visibility, and image text
alternatives.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


VIEWPORTS = (
    {"name": "mobile_375", "width": 375, "height": 812, "mobile": True},
    {"name": "mobile_390", "width": 390, "height": 844, "mobile": True},
    {"name": "desktop_1440", "width": 1440, "height": 1000, "mobile": False},
)


LAYOUT_SCRIPT = r"""
() => {
  const visible = (el) => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden'
      && rect.width > 0 && rect.height > 0;
  };
  const describe = (el) => {
    const id = el.id ? `#${el.id}` : '';
    const cls = [...el.classList].slice(0, 2).map((x) => `.${x}`).join('');
    return `${el.tagName.toLowerCase()}${id}${cls}`;
  };
  const rect = (el) => {
    const r = el.getBoundingClientRect();
    return {
      x: Math.round(r.x * 10) / 10,
      y: Math.round(r.y * 10) / 10,
      width: Math.round(r.width * 10) / 10,
      height: Math.round(r.height * 10) / 10,
      right: Math.round(r.right * 10) / 10,
      bottom: Math.round(r.bottom * 10) / 10,
    };
  };
  const interactive = [...document.querySelectorAll(
    'button, input, select, textarea, a[href], [role="button"], '
    + '.upload-area, .progress-bar, .timeline-item.seekable'
  )].filter(visible);
  const targets = interactive.map((el) => ({
    element: describe(el),
    role: el.getAttribute('role') || el.tagName.toLowerCase(),
    box: rect(el),
    text: (el.innerText || el.value || '').trim().slice(0, 80),
    ariaLabel: el.getAttribute('aria-label') || '',
    title: el.getAttribute('title') || '',
  }));
  const clippedText = [...document.querySelectorAll(
    'button, h1, h2, h3, p, small, .btn, .chip, .tool-btn, .status-value, '
    + '.time-display, .detection-item, .message'
  )].filter(visible).filter((el) => {
    if (el.children.length && el.innerText.trim().length > 100) return false;
    return el.scrollWidth > el.clientWidth + 2
      || el.scrollHeight > el.clientHeight + 2;
  }).map((el) => ({
    element: describe(el),
    text: (el.innerText || '').trim().slice(0, 120),
    client: [el.clientWidth, el.clientHeight],
    scroll: [el.scrollWidth, el.scrollHeight],
    overflow: getComputedStyle(el).overflow,
  }));
  const overlaps = [];
  const targetsForOverlap = targets.filter((item) => item.box.width > 0);
  for (let i = 0; i < targetsForOverlap.length; i += 1) {
    for (let j = i + 1; j < targetsForOverlap.length; j += 1) {
      const a = targetsForOverlap[i];
      const b = targetsForOverlap[j];
      const ix = Math.max(0, Math.min(a.box.right, b.box.right)
        - Math.max(a.box.x, b.box.x));
      const iy = Math.max(0, Math.min(a.box.bottom, b.box.bottom)
        - Math.max(a.box.y, b.box.y));
      const area = ix * iy;
      if (area > 16) {
        overlaps.push({
          first: a.element,
          second: b.element,
          overlap_area: Math.round(area),
        });
      }
    }
  }
  return {
    viewport: [window.innerWidth, window.innerHeight],
    document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
    horizontal_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
    targets,
    clipped_text: clippedText,
    overlaps,
    unlabeled_images: [...document.images].filter(
      (img) => !img.hasAttribute('alt')
    ).map((img) => img.id || img.src),
    main_landmark: Boolean(document.querySelector('main')),
    live_regions: document.querySelectorAll('[aria-live]').length,
    log_regions: document.querySelectorAll('[role="log"]').length,
  };
}
"""


def add_check(checks, name, passed, detail):
    checks.append({
        "name": name,
        "passed": bool(passed),
        "detail": detail,
    })


def audit_viewport(playwright, url, viewport):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": viewport["width"], "height": viewport["height"]},
        is_mobile=viewport["mobile"],
        has_touch=viewport["mobile"],
        device_scale_factor=1,
    )
    page = context.new_page()
    console_errors = []
    page_errors = []
    failed_requests = []
    page.on(
        "console",
        lambda msg: console_errors.append(msg.text)
        if msg.type == "error"
        else None,
    )
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.on(
        "requestfailed",
        lambda req: failed_requests.append({
            "url": req.url,
            "failure": req.failure,
        }),
    )
    response = page.goto(url, wait_until="load")
    page.wait_for_timeout(500)
    layout = page.evaluate(LAYOUT_SCRIPT)

    unnamed = []
    undersized = []
    for target in layout["targets"]:
        name = (
            target["ariaLabel"]
            or target["text"]
            or target["title"]
        ).strip()
        if not name:
            unnamed.append(target)
        box = target["box"]
        if viewport["mobile"] and (
            box["width"] < 44 or box["height"] < 44
        ):
            undersized.append(target)

    page.keyboard.press("Tab")
    focused = page.evaluate(
        """() => {
          const el = document.activeElement;
          if (!el || el === document.body) return null;
          const style = getComputedStyle(el);
          return {
            element: el.id ? `#${el.id}` : el.tagName.toLowerCase(),
            outline: `${style.outlineStyle} ${style.outlineWidth}`,
            boxShadow: style.boxShadow,
          };
        }"""
    )
    focus_visible = bool(
        focused
        and (
            focused["outline"] not in ("none 0px", "")
            or focused["boxShadow"] not in ("none", "")
        )
    )

    screenshot_dir = Path("evaluation/accessibility")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot = screenshot_dir / f"{viewport['name']}.png"
    page.screenshot(path=str(screenshot), full_page=True)
    browser.close()

    checks = []
    add_check(
        checks,
        f"{viewport['name']}_load",
        response is not None and response.ok,
        {
            "http_status": response.status if response else None,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "failed_requests": failed_requests,
        },
    )
    add_check(
        checks,
        f"{viewport['name']}_horizontal_overflow",
        not layout["horizontal_overflow"],
        {
            "viewport": layout["viewport"],
            "document": layout["document"],
        },
    )
    add_check(
        checks,
        f"{viewport['name']}_text_clipping",
        not layout["clipped_text"],
        layout["clipped_text"],
    )
    add_check(
        checks,
        f"{viewport['name']}_interactive_overlap",
        not layout["overlaps"],
        layout["overlaps"],
    )
    add_check(
        checks,
        f"{viewport['name']}_accessible_names",
        not unnamed,
        unnamed,
    )
    if viewport["mobile"]:
        add_check(
            checks,
            f"{viewport['name']}_touch_targets",
            not undersized,
            undersized,
        )
    add_check(
        checks,
        f"{viewport['name']}_keyboard_focus",
        focus_visible,
        {"focused": focused},
    )
    add_check(
        checks,
        f"{viewport['name']}_semantics",
        (
            layout["main_landmark"]
            and layout["live_regions"] >= 1
            and layout["log_regions"] >= 1
            and not layout["unlabeled_images"]
        ),
        {
            "main_landmark": layout["main_landmark"],
            "live_regions": layout["live_regions"],
            "log_regions": layout["log_regions"],
            "unlabeled_images": layout["unlabeled_images"],
        },
    )
    return {
        "name": viewport["name"],
        "checks": checks,
        "screenshot": str(screenshot.resolve()),
        "layout": layout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument(
        "--output",
        default="evaluation/responsive_accessibility_audit.json",
    )
    args = parser.parse_args()

    reports = []
    with sync_playwright() as playwright:
        for viewport in VIEWPORTS:
            reports.append(audit_viewport(playwright, args.url, viewport))

    checks = [
        check
        for viewport_report in reports
        for check in viewport_report["checks"]
    ]
    report = {
        "url": args.url,
        "viewports": reports,
        "checks": checks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = sum(check["passed"] is True for check in checks)
    failed = sum(check["passed"] is False for check in checks)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSummary: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
