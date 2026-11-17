-- PostgreSQL schema 1 of the API store (docs/api_design.md §7, D53 (4)). Applied once by
-- AnalysisRepository.create_schema() and recorded in schema_migration. Never edit a merged file:
-- a change is a new numbered file.
--
-- Equivalent to the SQLite tables it replaces (SDD v2.0 §5.2): the same columns, keys and CHECKs,
-- with three type changes and one added column:
--   * confidence is DOUBLE PRECISION (PostgreSQL REAL is 4 bytes and would round the value);
--   * is_mock and disabled are BOOLEAN instead of INTEGER 0/1;
--   * account_id is an identity column instead of SQLite's rowid alias;
--   * analysis_record.seq (identity) replaces SQLite's rowid as the tie-break of records stored in
--     the same second. It is internal and never returned by the API.
-- Times stay TEXT, UTC, ISO 8601 with Z, compared as strings; the database uses the C collation.
-- Never stored: audio, the original file name, the client IP address, passwords in clear, session
-- IDs in clear, the typed username of a failed sign-in (only its HMAC). No table has a column for them.

CREATE TABLE analysis_record (
    request_id      TEXT             PRIMARY KEY,
    created_at      TEXT             NOT NULL,
    prediction      TEXT             NOT NULL CHECK (prediction IN ('synthetic', 'uncertain', 'authentic')),
    confidence      DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    risk_level      TEXT             NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    model_version   TEXT             NOT NULL,
    profile_version TEXT             NOT NULL,
    policy_version  TEXT             NOT NULL,
    notice          TEXT             NOT NULL,
    processing_ms   INTEGER          NOT NULL CHECK (processing_ms >= 0),
    is_mock         BOOLEAN          NOT NULL,
    seq             BIGINT           NOT NULL GENERATED ALWAYS AS IDENTITY UNIQUE
);

-- Usernames: rules in api/auth.py; the CHECK keeps the stored form lower case and in length.
CREATE TABLE account (
    account_id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username            TEXT    NOT NULL UNIQUE
                        CHECK (username = lower(username) AND length(username) BETWEEN 3 AND 64),
    password_hash       TEXT    NOT NULL CHECK (password_hash LIKE '$argon2id$%'),
    disabled            BOOLEAN NOT NULL,
    created_at          TEXT    NOT NULL
                        CHECK (created_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'),
    password_changed_at TEXT    NOT NULL
                        CHECK (password_changed_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$')
);

-- note_max_chars is config (§9), so the API enforces it; a CHECK here would need a migration per change.
-- updated_by_account_id is null for notes saved before sign-in existed (§7).
CREATE TABLE analysis_note (
    request_id            TEXT    NOT NULL PRIMARY KEY REFERENCES analysis_record (request_id),
    text                  TEXT    NOT NULL CHECK (length(trim(text)) > 0),
    updated_at            TEXT    NOT NULL
                          CHECK (updated_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'),
    updated_by_account_id INTEGER REFERENCES account (account_id)
);

-- Written by sign-in; no IP or user-agent column (§4.8 privacy).
CREATE TABLE session (
    session_hash TEXT    PRIMARY KEY CHECK (session_hash ~ '^[0-9a-f]{64}$'),
    account_id   INTEGER NOT NULL REFERENCES account (account_id),
    created_at   TEXT    NOT NULL CHECK (created_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'),
    last_seen_at TEXT    NOT NULL CHECK (last_seen_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'),
    expires_at   TEXT    NOT NULL CHECK (expires_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$')
);

CREATE TABLE sign_in_failure (
    username_hmac   TEXT    PRIMARY KEY CHECK (username_hmac ~ '^[0-9a-f]{64}$'),
    failures        INTEGER NOT NULL CHECK (failures >= 0),
    last_failure_at TEXT    NOT NULL
                    CHECK (last_failure_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'),
    locked_until    TEXT
                    CHECK (locked_until IS NULL
                           OR locked_until ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$')
);
