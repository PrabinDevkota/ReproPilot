#!/usr/bin/env python3
"""Smoke ReproPilot HTTP endpoints."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LiveClient:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def request(self, method: str, path: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, str]:
        req = urllib.request.Request(urljoin(self.base + "/", path.lstrip("/")), data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, resp.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")


class LocalClient:
    def __init__(self) -> None:
        from fastapi.testclient import TestClient
        from server.app import app

        self._client = TestClient(app, raise_server_exceptions=True)

    def request(self, method: str, path: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, str]:
        kwargs: dict[str, Any] = {}
        if data is not None:
            kwargs["content"] = data
            kwargs["headers"] = headers or {}
        resp = self._client.request(method, path, **kwargs)
        return resp.status_code, resp.text


def main() -> int:
    parser = argparse.ArgumentParser(description="ReproPilot HTTP endpoint smoke.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    client: LiveClient | LocalClient = LocalClient() if args.local else LiveClient(args.url)
    for path in ("/health", "/metadata", "/state", "/schema", "/openapi.json"):
        code, _ = client.request("GET", path)
        print(f"GET {path}: {code}")
        if not 200 <= code < 300:
            return 1
    headers = {"Content-Type": "application/json"}
    code, body = client.request("POST", "/reset", data=b"{}", headers=headers)
    print(f"POST /reset: {code}")
    if code != 200 or "REPROPILOT RESEARCH AUDIT BRIEFING" not in body:
        return 1
    payload = {"action": {"action_type": "run_split_check", "target_id": "claim_001", "explanation": "verify split"}}
    code, body = client.request("POST", "/step", data=json.dumps(payload).encode(), headers=headers)
    print(f"POST /step run_split_check: {code}")
    return 0 if code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
