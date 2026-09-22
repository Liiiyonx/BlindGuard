# -*- coding: utf-8 -*-
"""Authentication and server binding helpers for BlindGuard."""

import ipaddress
import secrets
from typing import Mapping, Optional

AUTH_HEADER = "X-BlindGuard-Token"


def is_loopback_host(host: str) -> bool:
    """Return True when a configured bind address only exposes this machine."""
    value = (host or "").strip().strip("[]").lower()
    if value in {"localhost", "::1"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def validate_server_binding(host: str, auth_token: Optional[str]) -> None:
    """Refuse a non-loopback bind when no authentication token is configured."""
    if not is_loopback_host(host) and not (auth_token or "").strip():
        raise RuntimeError(
            "安全配置错误：server.host 不是本机回环地址，但未配置访问令牌。"
            "请设置环境变量 BLINDGUARD_SERVER_TOKEN，或配置 server.auth_token。"
        )


def extract_token(headers: Mapping[str, str],
                  query_args: Optional[Mapping[str, str]] = None) -> str:
    """Read a token from the dedicated header, bearer header, or query string."""
    token = (headers.get(AUTH_HEADER) or "").strip()
    if token:
        return token

    authorization = (headers.get("Authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()

    if query_args is not None:
        return (query_args.get("token") or "").strip()
    return ""


def token_is_valid(provided_token: str, expected_token: Optional[str]) -> bool:
    """Constant-time token comparison with explicit empty-token handling."""
    expected = (expected_token or "").strip()
    if not expected:
        return True
    if not provided_token:
        return False
    return secrets.compare_digest(provided_token, expected)


def is_protected_path(path: str) -> bool:
    """Protect the page, all APIs, and the MJPEG feed."""
    return path == "/" or path == "/video_feed" or path.startswith("/api/")
