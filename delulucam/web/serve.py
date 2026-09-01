#!/usr/bin/env python3
"""Serve delulucam's cloud-avatar web app locally and open it in the browser.

A plain file:// URL often blocks getUserMedia (camera) access in Chrome/Safari;
serving over http://localhost sidesteps that (localhost counts as a secure
context) without needing a real TLS certificate.

Usage:
    python3 delulucam/web/serve.py [port]   # default port 8420
"""

import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8420
WEB_DIR = Path(__file__).parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)


def main():
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"[web] serving {WEB_DIR} at {url}")
        print("[web] point OBS's Browser Source at this same URL to feed your virtual camera")
        print("[web] Ctrl-C to stop")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[web] stopped")


if __name__ == "__main__":
    main()
