import base64
import re
import secrets

import pytest

from api.auth import (HMAC_KEY_ENV, normalize_username, password_hasher, password_problem, signin_key_from_env,
                      username_hmac)

PASSWORD = "correct horse battery staple"


def test_hasher_is_argon2id_rfc_9106_low_memory_with_a_16_byte_salt():
    stored = password_hasher().hash(PASSWORD)
    assert stored.startswith("$argon2id$v=19$m=65536,t=3,p=4$")
    salt = stored.split("$")[4]
    assert len(base64.b64decode(salt + "=" * (-len(salt) % 4))) == 16
    assert password_hasher().verify(stored, PASSWORD)
    assert not password_hasher().check_needs_rehash(stored)


def test_signin_key_is_read_from_the_environment(monkeypatch):
    key = secrets.token_bytes(32)
    monkeypatch.setenv(HMAC_KEY_ENV, key.hex())
    assert signin_key_from_env() == key


@pytest.mark.parametrize("value", [None, "", "not-hex-" * 10, secrets.token_bytes(31).hex()],
                         ids=["missing", "empty", "not hex", "31 bytes"])
def test_signin_key_is_refused_without_showing_its_value(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(HMAC_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(HMAC_KEY_ENV, value)
    with pytest.raises(ValueError, match=HMAC_KEY_ENV) as caught:
        signin_key_from_env()
    if value:
        assert value not in str(caught.value)


def test_username_hmac_ignores_case_and_depends_on_the_key():
    key = secrets.token_bytes(32)
    digest = username_hmac(key, "Analyst01")
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == username_hmac(key, "analyst01")
    assert digest != username_hmac(secrets.token_bytes(32), "analyst01")
    assert "analyst01" not in digest


@pytest.mark.parametrize("text, stored", [
    ("Analyst01", "analyst01"), ("abc", "abc"), ("a.b_c-9", "a.b_c-9"), ("x" * 64, "x" * 64)])
def test_normalize_username_accepts_and_lower_cases(text, stored):
    assert normalize_username(text) == stored


@pytest.mark.parametrize("text", ["ab", "x" * 65, "analyst 01", "analyst@01", "analÿst", ""])
def test_normalize_username_refuses_other_forms(text):
    with pytest.raises(ValueError):
        normalize_username(text)


def test_password_length_is_counted_in_code_points():
    assert password_problem("a" * 15, "analyst01", 15, 128) is None
    assert password_problem("😀" * 15, "analyst01", 15, 128) is None  # 15 code points, 60 UTF-8 bytes
    assert "at least 15" in password_problem("😀" * 14, "analyst01", 15, 128)
    assert password_problem("a" * 128, "analyst01", 15, 128) is None
    assert "at most 128" in password_problem("a" * 129, "analyst01", 15, 128)


def test_password_has_no_composition_rules():
    assert password_problem("all lower case words here", "analyst01", 15, 128) is None


@pytest.mark.parametrize("password", [
    "my-ANALYST01-password", "VSD-is-the-best-tool", "my Voice is my password", "please do not SPOOF me"])
def test_password_must_not_contain_the_username_or_a_service_word(password):
    problem = password_problem(password, "analyst01", 15, 128)
    assert problem is not None
    assert password not in problem
