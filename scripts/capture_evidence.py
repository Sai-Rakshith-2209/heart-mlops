"""Capture deployment evidence screenshots with a headless browser.

- live pages: Grafana dashboard, Prometheus targets, API Swagger docs (via port-forwards)
- terminal output: renders the collected kubectl/curl transcripts as terminal-style PNGs

Usage: GRAFANA_PASSWORD=... python scripts/capture_evidence.py --out deploy-evidence
"""

from __future__ import annotations

import argparse
import base64
import html
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

TERMINAL_FILES = {
    "k8s_deploy.txt": "06_k8s_deploy_and_predict.png",
    "pod_logs.txt": "10_pod_json_logs.png",
    "prometheus_targets.txt": "08b_prometheus_targets_api.png",
    "load_test.txt": "09b_load_test.png",
}

# Pinned locale/timezone: the Prometheus UI crashes on environment-derived locales such as
# "en-US@posix" (blank page), and fixed settings make screenshots identical on any machine.
BROWSER_SETTINGS = {"locale": "en-US", "timezone_id": "Asia/Kolkata"}


def terminal_html(title: str, text: str) -> str:
    return f"""<html><body style="margin:0;background:#0d1117">
<div style="font:600 13px system-ui;color:#c9d1d9;background:#161b22;padding:8px 14px">{
        html.escape(title)
    }</div>
<pre style="margin:0;padding:14px;color:#e6edf3;font:13px/1.45 'DejaVu Sans Mono',monospace;
white-space:pre">{html.escape(text)}</pre></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deploy-evidence")
    ap.add_argument("--grafana", default="http://localhost:3000")
    ap.add_argument("--prometheus", default="http://localhost:9090")
    ap.add_argument("--api", default="http://localhost:8080")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    auth = base64.b64encode(f"admin:{os.environ['GRAFANA_PASSWORD']}".encode()).decode()

    failures: list[str] = []

    def shoot(name: str, action) -> None:
        """Run one capture; record a failure instead of aborting the others."""
        try:
            action()
        except (
            Exception
        ) as exc:  # noqa: BLE001 - report every failed capture at the end
            failures.append(
                f"{name}: {exc.__class__.__name__}: {str(exc).splitlines()[0]}"
            )

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # Grafana accepts HTTP basic auth on every request, so no login form is needed.
        grafana = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            extra_http_headers={"Authorization": f"Basic {auth}"},
            **BROWSER_SETTINGS,
        )
        plain = browser.new_context(
            viewport={"width": 1600, "height": 900}, **BROWSER_SETTINGS
        )

        def grafana_dashboard() -> None:
            page = grafana.new_page()
            page.goto(
                f"{a.grafana}/d/heart-api/heart-disease-api?orgId=1&from=now-15m&to=now&kiosk",
                wait_until="load",
            )
            # Live UIs poll continuously, so "network idle" never comes: wait for content.
            page.get_by_text("Requests / s").filter(visible=True).first.wait_for(
                timeout=60000
            )
            page.wait_for_timeout(10000)  # let every panel finish querying Prometheus
            page.screenshot(path=out / "09_grafana_dashboard.png", full_page=True)

        def prometheus_targets() -> None:
            page = plain.new_page()
            page.goto(f"{a.prometheus}/targets", wait_until="load")
            page.get_by_text("/metrics").filter(visible=True).first.wait_for(
                timeout=60000
            )
            page.wait_for_timeout(1000)
            page.screenshot(path=out / "08_prometheus_targets.png", full_page=True)

        def api_docs() -> None:
            page = plain.new_page()
            page.goto(f"{a.api}/docs", wait_until="load")
            # Expand the POST /predict operation so its request schema is visible.
            page.locator(".opblock-post .opblock-summary").first.click(timeout=30000)
            page.wait_for_timeout(1500)
            page.screenshot(path=out / "03_api_docs.png", full_page=True)

        shoot("grafana", grafana_dashboard)
        shoot("prometheus", prometheus_targets)
        shoot("api-docs", api_docs)

        page = plain.new_page()
        for src, dst in TERMINAL_FILES.items():
            f = out / src
            if f.exists():
                page.set_content(terminal_html(src, f.read_text()))
                shoot(
                    dst, lambda dst=dst: page.screenshot(path=out / dst, full_page=True)
                )
        browser.close()

    print("screenshots:", sorted(x.name for x in out.glob("*.png")))
    if failures:
        raise SystemExit("capture failures:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
