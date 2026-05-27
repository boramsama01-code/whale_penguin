"""
Simple SQLite-based user authentication and data store.
No external dependencies — uses only Python built-ins.
"""
import sqlite3
import hashlib
import secrets
import json
import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "user_data.db")
_SALT = "krx_whale_tracker_v1_salt"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """DB 및 테이블 초기화 — 앱 시작 시 1회 호출."""
    try:
        with _conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                created_at INTEGER DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS user_data (
                user_id INTEGER PRIMARY KEY,
                portfolio TEXT NOT NULL DEFAULT '[]',
                watchlist TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)
        logger.info("user_store DB 초기화 완료: %s", DB_PATH)
    except Exception as e:
        logger.error("user_store DB 초기화 실패: %s", e)


def _hash(password: str) -> str:
    return hashlib.sha256(f"{_SALT}{password}".encode("utf-8")).hexdigest()


def register_user(username: str, password: str, display_name: str = "") -> Optional[dict]:
    """새 사용자 등록. 성공 시 user dict 반환, 중복 시 None."""
    try:
        uname = username.strip().lower()
        if len(uname) < 3:
            return None
        ph = _hash(password)
        dname = (display_name or username).strip()[:20]
        with _conn() as c:
            c.execute(
                "INSERT INTO users (username, password_hash, display_name) VALUES (?, ?, ?)",
                (uname, ph, dname),
            )
            uid = c.lastrowid
            c.execute("INSERT OR IGNORE INTO user_data (user_id) VALUES (?)", (uid,))
        return {"id": uid, "username": uname, "display_name": dname}
    except sqlite3.IntegrityError:
        return None
    except Exception as e:
        logger.error("register_user 오류: %s", e)
        return None


def login_user(username: str, password: str) -> Optional[dict]:
    """로그인. 성공 시 {token, id, username, display_name} 반환."""
    try:
        ph = _hash(password)
        with _conn() as c:
            row = c.execute(
                "SELECT id, username, display_name FROM users WHERE username=? AND password_hash=?",
                (username.strip().lower(), ph),
            ).fetchone()
            if not row:
                return None
            token = secrets.token_urlsafe(32)
            expires = int(time.time()) + 86400 * 30  # 30일
            c.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, row["id"], expires),
            )
        return {
            "token": token,
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
        }
    except Exception as e:
        logger.error("login_user 오류: %s", e)
        return None


def get_user_by_token(token: str) -> Optional[dict]:
    """토큰으로 사용자 조회. 만료되었거나 없으면 None."""
    if not token:
        return None
    try:
        with _conn() as c:
            row = c.execute(
                """SELECT u.id, u.username, u.display_name
                   FROM sessions s JOIN users u ON u.id = s.user_id
                   WHERE s.token=? AND s.expires_at > ?""",
                (token, int(time.time())),
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.debug("get_user_by_token 오류: %s", e)
        return None


def logout_token(token: str):
    """세션 토큰 삭제."""
    try:
        with _conn() as c:
            c.execute("DELETE FROM sessions WHERE token=?", (token,))
    except Exception as e:
        logger.debug("logout_token 오류: %s", e)


def get_user_data(user_id: int) -> dict:
    """사용자 포트폴리오 + 즐겨찾기 반환."""
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT portfolio, watchlist FROM user_data WHERE user_id=?", (user_id,)
            ).fetchone()
        if not row:
            return {"portfolio": [], "watchlist": []}
        return {
            "portfolio": json.loads(row["portfolio"] or "[]"),
            "watchlist": json.loads(row["watchlist"] or "[]"),
        }
    except Exception as e:
        logger.error("get_user_data 오류: %s", e)
        return {"portfolio": [], "watchlist": []}


def save_portfolio(user_id: int, portfolio: list):
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO user_data (user_id, portfolio) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET portfolio=excluded.portfolio",
                (user_id, json.dumps(portfolio, ensure_ascii=False)),
            )
    except Exception as e:
        logger.error("save_portfolio 오류: %s", e)


def save_watchlist(user_id: int, watchlist: list):
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO user_data (user_id, watchlist) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET watchlist=excluded.watchlist",
                (user_id, json.dumps(watchlist, ensure_ascii=False)),
            )
    except Exception as e:
        logger.error("save_watchlist 오류: %s", e)
