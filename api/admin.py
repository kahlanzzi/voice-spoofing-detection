"""Account administration (docs/api_design.md §4.8): python -m api.admin <subcommand> <username>

Subcommands: create-account, set-password, disable, enable, unlock. Accounts exist only through
this command; there is no signup or password-change endpoint. A new password is typed twice at a
hidden prompt (getpass), never given on the command line or in a file, and is never printed.
set-password and disable end all of the account's sessions. unlock resets the failure count of a
typed username, known or not, and needs VSD_SIGNIN_HMAC_KEY in the environment.

The database and the password limits come from the API config ($VSD_API_CONFIG or
configs/api.yaml). Exit codes: 0 done, 1 refused (one line on stderr), 2 usage error.
"""
from __future__ import annotations

import argparse
import getpass
import sys

from api.auth import normalize_username, password_hasher, password_problem, signin_key_from_env, username_hmac
from api.config import ApiConfig, load_api_config, resolve_path
from api.repository import AccountRepository, AnalysisRepository, DuplicateAccount
from api.routers.status import utc_now_iso

SUBCOMMANDS = {
    "create-account": "Create an enabled account and set its password.",
    "set-password": "Set a new password; ends all of the account's sessions.",
    "disable": "Disable the account; ends all of its sessions.",
    "enable": "Enable a disabled account.",
    "unlock": "Reset the failed sign-in count and lock of a username.",
}


class Refused(Exception):
    """A refused admin action; its message is safe to print (never a password)."""


def read_new_password(username: str, config: ApiConfig) -> str:
    """The new password, typed twice at a hidden prompt and checked against the §4.8 rules."""
    password = getpass.getpass("New password: ")
    if getpass.getpass("Repeat the password: ") != password:
        raise Refused("The two passwords differ. Nothing was changed.")
    problem = password_problem(password, username, config.password_min_chars, config.password_max_chars)
    if problem:
        raise Refused(f"{problem} Nothing was changed.")
    return password


def run(command: str, username: str, config: ApiConfig) -> str:
    """Do one admin action and return the line to print; raise Refused if it cannot be done."""
    if command == "unlock":
        # Any typed text, account or not: lockout counts every submitted username (§4.8).
        try:
            key = signin_key_from_env()
        except ValueError as exc:
            raise Refused(str(exc)) from None
        AnalysisRepository(resolve_path(config.database_path)).create_schema()
        held = AccountRepository(resolve_path(config.database_path)).clear_failures(username_hmac(key, username))
        return f"Username {username} unlocked." if held else f"Username {username} had no failed sign-ins."
    try:
        username = normalize_username(username)
    except ValueError as exc:
        raise Refused(str(exc)) from None
    database = resolve_path(config.database_path)
    AnalysisRepository(database).create_schema()
    accounts = AccountRepository(database)

    if command == "create-account":
        if accounts.get_account(username) is not None:
            raise Refused(f"Account {username} already exists.")
        password_hash = password_hasher().hash(read_new_password(username, config))
        try:
            accounts.create_account(username, password_hash, utc_now_iso())
        except DuplicateAccount:
            raise Refused(f"Account {username} already exists.") from None
        return f"Account {username} created."
    if command == "set-password":
        if accounts.get_account(username) is None:
            raise Refused(f"No account {username}.")
        password_hash = password_hasher().hash(read_new_password(username, config))
        ended = accounts.set_password(username, password_hash, utc_now_iso())
        if ended is None:
            raise Refused(f"No account {username}.")
        return f"Password of {username} set; {ended} session(s) ended."
    if command in ("disable", "enable"):
        ended = accounts.set_disabled(username, command == "disable")
        if ended is None:
            raise Refused(f"No account {username}.")
        if command == "enable":
            return f"Account {username} enabled."
        return f"Account {username} disabled; {ended} session(s) ended."
    raise Refused(f"Unknown subcommand {command}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m api.admin", description="Manage analyst accounts (§4.8).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in SUBCOMMANDS.items():
        commands.add_parser(name, help=help_text).add_argument("username")
    args = parser.parse_args(argv)
    try:
        print(run(args.command, args.username, load_api_config()))
    except Refused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
