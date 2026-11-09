"""Sign-in building blocks (docs/api_design.md §4.8, D47, ADR 0011): the argon2id hasher, the
VSD_SIGNIN_HMAC_KEY secret, the username HMAC behind sign_in_failure, and the username and password
rules used by the admin command.

The key comes from the environment only, never from configs/ or the repo, and no error message
carries it. Password checks answer with fixed sentences that never contain the password.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re

from argon2 import PasswordHasher, profiles

HMAC_KEY_ENV = "VSD_SIGNIN_HMAC_KEY"
HMAC_KEY_MIN_BYTES = 32
USERNAME = re.compile(r"[a-z0-9._-]{3,64}")
# NIST SP 800-63B-4 §3.1.1.2 context-specific words: the service name and its derivatives.
CONTEXT_WORDS = ("vsd", "voice", "spoof")


def password_hasher() -> PasswordHasher:
    """argon2id with the RFC 9106 low-memory profile: t=3, m=64 MiB, p=4, 16-byte salt (§4.8)."""
    return PasswordHasher.from_parameters(profiles.RFC_9106_LOW_MEMORY)


def signin_key_from_env() -> bytes:
    """The HMAC key from $VSD_SIGNIN_HMAC_KEY: hex, at least 32 bytes. Raise ValueError otherwise."""
    value = os.environ.get(HMAC_KEY_ENV, "")
    if not value:
        raise ValueError(f"{HMAC_KEY_ENV} is not set")
    try:
        key = bytes.fromhex(value)
    except ValueError:
        raise ValueError(f"{HMAC_KEY_ENV} must be hex") from None
    if len(key) < HMAC_KEY_MIN_BYTES:
        raise ValueError(f"{HMAC_KEY_ENV} must hold at least {HMAC_KEY_MIN_BYTES} bytes")
    return key


def username_hmac(key: bytes, username: str) -> str:
    """HMAC-SHA-256 hex of the lower-case typed username: the sign_in_failure key (§7)."""
    return hmac.new(key, username.lower().encode("utf-8"), hashlib.sha256).hexdigest()


def normalize_username(text: str) -> str:
    """The stored form of a username: lower case, 3 to 64 of a-z 0-9 . _ - ; ValueError otherwise."""
    username = text.lower()
    if not USERNAME.fullmatch(username):
        raise ValueError("A username has 3 to 64 characters from a-z 0-9 . _ -")
    return username


def password_problem(password: str, username: str, min_chars: int, max_chars: int) -> str | None:
    """Why a new password is refused, or None. Length in code points, no composition rules; it must
    not contain the username or a service word, ignoring case (NIST SP 800-63B-4 §3.1.1.2)."""
    if len(password) < min_chars:
        return f"The password must have at least {min_chars} characters."
    if len(password) > max_chars:
        return f"The password must have at most {max_chars} characters."
    folded = password.casefold()
    if username.casefold() in folded:
        return "The password must not contain the username."
    if any(word in folded for word in CONTEXT_WORDS):
        return "The password must not contain the service name (vsd, voice, spoof)."
    return None
