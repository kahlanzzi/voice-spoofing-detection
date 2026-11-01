"""deploy/nginx.conf (Stage 17 Step 8b): routing, error pass-through, sign-in rate limit, no client address.

CI has no nginx, so the file is read with a small parser (directives, blocks, quotes, comments) and the
directives are checked against docs/api_design.md and configs/api.yaml. `nginx -t` is part of the manual run
(docs/setup.md).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from api.config import load_api_config
from api.intake import body_limit

REPO = Path(__file__).resolve().parents[2]
TEXT = (REPO / "deploy" / "nginx.conf").read_text(encoding="utf-8")
ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}
UNITS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


class Node:
    def __init__(self, name: str, args: list[str], children: list[Node] | None):
        self.name, self.args, self.children = name, args, children

    def find(self, name: str) -> list[Node]:
        return [c for c in self.children or [] if c.name == name]

    def walk(self):
        for child in self.children or []:
            yield child
            yield from child.walk()


def tokenize(text: str) -> list[str]:
    tokens, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch == "#":
            i = text.find("\n", i) if "\n" in text[i:] else len(text)
        elif ch in "{};":
            tokens.append(ch)
            i += 1
        elif ch in "'\"":
            out, i = [], i + 1
            while text[i] != ch:
                if text[i] == "\\":
                    i += 1
                    out.append(ESCAPES.get(text[i], text[i]))
                else:
                    out.append(text[i])
                i += 1
            tokens.append("".join(out))
            i += 1
        else:
            match = re.compile(r"[^\s{};'\"]+").match(text, i)
            tokens.append(match.group())
            i = match.end()
    return tokens


def parse(tokens: list[str]) -> Node:
    """Nested Nodes; a block directive has children, a simple one has None."""
    stack = [Node("", [], [])]
    words: list[str] = []
    for token in tokens:
        if token == ";":
            stack[-1].children.append(Node(words[0], words[1:], None))
            words = []
        elif token == "{":
            node = Node(words[0], words[1:], [])
            stack[-1].children.append(node)
            stack.append(node)
            words = []
        elif token == "}":
            assert not words, f"unterminated directive {words}"
            stack.pop()
        else:
            words.append(token)
    assert len(stack) == 1 and not words, "unbalanced braces"
    return stack[0]


ROOT = parse(tokenize(TEXT))
(HTTP,) = ROOT.find("http")
(SERVER,) = HTTP.find("server")
LOCATIONS = {" ".join(n.args): n for n in SERVER.find("location")}


def size_bytes(value: str) -> int:
    match = re.fullmatch(r"(\d+)([kmg]?)", value.lower())
    return int(match.group(1)) * UNITS[match.group(2)]


def seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+)s", value)
    return int(match.group(1))


def section6_codes() -> set[tuple[str, int]]:
    """(code, HTTP status) rows of the docs/api_design.md §6 table."""
    doc = (REPO / "docs" / "api_design.md").read_text(encoding="utf-8")
    section = doc.split("## 6. Errors", 1)[1].split("\n## ", 1)[0]
    return {(c, int(s)) for c, s in re.findall(r"^\| `([a-z_]+)` \| (\d{3}) \|", section, re.M)}


# Error pass-through (§6)

def test_api_errors_are_not_intercepted():
    assert not any(n.name == "proxy_intercept_errors" for n in ROOT.walk())


def test_error_pages_cover_only_nginx_own_answers():
    codes = {int(a) for n in SERVER.find("error_page") for a in n.args if a.isdigit()}
    assert codes == {404, 413, 429, 502, 504}  # never 500 or 503: those come from the API unchanged
    assert not [n for n in ROOT.walk() if n.name == "error_page" and n not in SERVER.find("error_page")]
    (gateway,) = [n for n in SERVER.find("error_page") if "502" in n.args]
    assert gateway.args == ["502", "504", "=503", "@unavailable"]


def test_nginx_answers_use_the_section6_shape_and_codes():
    rows = section6_codes()
    targets = {n.args[-1] for n in SERVER.find("error_page")}
    assert targets == {"@not_found", "@too_large", "@too_many", "@unavailable"}
    expected = {"@not_found": "not_found", "@too_large": "file_too_large", "@too_many": "too_many_requests",
                "@unavailable": "analysis_unavailable"}
    for target in targets:
        location = LOCATIONS[target]
        assert location.find("default_type")[0].args == ["application/json"]
        (ret,) = location.find("return")
        status, body = int(ret.args[0]), json.loads(ret.args[1])
        assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "request_id"}
        assert body["error"]["request_id"] is None
        assert body["error"]["code"] == expected[target]
        assert (body["error"]["code"], status) in rows


def test_body_limit_and_timeouts_sit_above_the_api_limits():
    config = load_api_config()
    (limit,) = SERVER.find("client_max_body_size")
    assert size_bytes(limit.args[0]) > body_limit(config)  # the API's own 413 answers first
    for name in ("proxy_read_timeout", "proxy_send_timeout"):
        (timeout,) = SERVER.find(name)
        assert seconds(timeout.args[0]) > config.analysis_timeout_s  # the API's own 504 answers first


# Sign-in rate limit (§4.8, §9)

def test_rate_limit_keys_only_post_by_client_address():
    (mapping,) = HTTP.find("map")
    assert mapping.args == ["$request_method", "$signin_limit_key"]
    assert {n.name: n.args for n in mapping.children} == {"POST": ["$binary_remote_addr"], "default": [""]}
    (zone,) = HTTP.find("limit_req_zone")
    assert zone.args == ["$signin_limit_key", "zone=signin:1m", "rate=10r/m"]
    assert HTTP.find("limit_req_status")[0].args == ["429"]


def test_rate_limit_applies_only_to_the_session_route_with_burst_and_nodelay():
    limited = [loc for loc, node in LOCATIONS.items() if node.find("limit_req")]
    assert limited == ["= /api/session"]
    assert LOCATIONS["= /api/session"].find("limit_req")[0].args == ["zone=signin", "burst=5", "nodelay"]
    assert [n for n in ROOT.walk() if n.name == "limit_req"] == LOCATIONS["= /api/session"].find("limit_req")


# No client address (§1, §7)

def test_no_client_address_or_request_header_variable_outside_the_rate_limit_map():
    for banned in ("$remote_addr", "$http_", "$proxy_add_x_forwarded_for", "$realip", "$proxy_protocol_addr",
                   "$request ", "$request_uri", "$args", "$query_string"):
        assert banned not in TEXT.replace("$binary_remote_addr", ""), banned
    uses = [n for n in ROOT.walk() if "$binary_remote_addr" in n.args]
    assert len(uses) == 1 and uses[0].name == "POST"


def test_logs_use_the_address_free_format_and_errors_only_at_crit():
    (fmt,) = HTTP.find("log_format")
    assert fmt.args[0] == "vsd"
    access = [n for n in ROOT.walk() if n.name == "access_log"]
    assert access and all(n.args[-1] == "vsd" for n in access)
    errors = [n for n in ROOT.walk() if n.name == "error_log"]
    assert errors and all(n.args[-1] == "crit" for n in errors)
    assert not ROOT.find("include") and HTTP.find("include")[0].args == ["/etc/nginx/mime.types"]
    assert all("conf.d" not in a for n in ROOT.walk() if n.name == "include" for a in n.args)
    assert HTTP.find("server_tokens")[0].args == ["off"]


def test_forwarded_headers_are_cleared_towards_the_api():
    cleared = {n.args[0]: n.args[1] for n in SERVER.find("proxy_set_header")}
    assert cleared == {"X-Forwarded-For": "", "X-Real-IP": "", "Forwarded": ""}
    # A location with its own proxy_set_header would not inherit these.
    assert not any(node.find("proxy_set_header") for node in LOCATIONS.values())


# Routing (§2)

def test_only_api_paths_are_proxied_and_root_is_the_placeholder():
    proxied = {loc: node.find("proxy_pass")[0].args for loc, node in LOCATIONS.items() if node.find("proxy_pass")}
    assert proxied == {"= /api/session": ["http://api:8000"], "/api/": ["http://api:8000"]}
    (ret,) = LOCATIONS["= /"].find("return")
    assert ret.args[0] == "200"
    assert LOCATIONS["/"].find("return")[0].args == ["404"]


def test_fastapi_docs_routes_are_not_routed():
    for path in ("/docs", "/redoc", "/openapi.json"):
        matching = [loc for loc in LOCATIONS if not loc.startswith("@")
                    and (loc == f"= {path}" or (" " not in loc and path.startswith(loc)))]
        assert matching == ["/"], (path, matching)  # only the catch-all 404


def test_parser_reads_quotes_escapes_and_comments():
    root = parse(tokenize("a 'x;y' \"q\\\"\"; # c { ;\nb { c '' ; }\n"))
    assert [(n.name, n.args) for n in root.children] == [("a", ["x;y", 'q"']), ("b", [])]
    assert root.find("b")[0].children[0].args == [""]
