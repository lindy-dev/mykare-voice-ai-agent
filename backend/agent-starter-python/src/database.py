import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import phonenumbers
from phonenumbers import NumberParseException

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "voice_agent.db")

logger = logging.getLogger(__name__)

_initialized = False
_init_lock = asyncio.Lock()
_upsert_lock = asyncio.Lock()


def _row_to_dict(row: tuple, columns: list[str]) -> dict[str, Any]:
    return dict(zip(columns, row))


def validate_phone(phone: str) -> tuple[bool, str]:
    """Validate a phone number and return (is_valid, result).

    On success, result is the normalized E.164 phone number.
    On failure, result is an error message.

    All phone numbers must include a + prefix with country code.
    """
    if not phone or not phone.strip():
        logger.warning("Phone validation failed: empty phone number")
        return False, "Phone number is empty"

    phone = phone.strip()

    if not phone.startswith("+"):
        logger.warning(
            "Phone validation failed: missing + prefix for phone '%s'", phone
        )
        return (
            False,
            "Phone number must include a country code with + prefix (e.g., +1 234 567 8900)",
        )

    try:
        parsed = phonenumbers.parse(phone, region=None)
    except NumberParseException as e:
        logger.warning("Phone validation failed: cannot parse '%s': %s", phone, e)
        return False, "Phone number is not in a valid format"

    if not phonenumbers.is_valid_number(parsed):
        logger.warning("Phone validation failed: invalid number '%s'", phone)
        return False, "Phone number is not valid"

    normalized = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    logger.info("Phone validated: '%s' -> '%s'", phone, normalized)
    return True, normalized


async def _ensure_init() -> None:
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        os.makedirs(DB_DIR, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.commit()
        _initialized = True


async def get_user_by_phone(phone: str) -> dict[str, Any] | None:
    await _ensure_init()
    async with (
        aiosqlite.connect(DB_PATH) as db,
        db.execute(
            "SELECT id, name, phone, created_at, updated_at FROM users WHERE phone = ?",
            (phone,),
        ) as cursor,
    ):
        row = await cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _row_to_dict(row, columns)


async def upsert_user(name: str, phone: str) -> dict[str, Any]:
    valid, result = validate_phone(phone)
    if not valid:
        raise ValueError(result)
    phone = result

    await _ensure_init()
    async with _upsert_lock:
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT name FROM users WHERE phone = ?",
                (phone,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is not None:
                existing_name = row[0]
                if existing_name != name:
                    await db.execute(
                        "UPDATE users SET name = ?, updated_at = ? WHERE phone = ?",
                        (name, now, phone),
                    )
                    await db.commit()
            else:
                user_id = str(uuid.uuid4())
                await db.execute(
                    "INSERT INTO users (id, name, phone, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, name, phone, now, now),
                )
                await db.commit()

            async with db.execute(
                "SELECT id, name, phone, created_at, updated_at FROM users WHERE phone = ?",
                (phone,),
            ) as cursor:
                result_row = await cursor.fetchone()
                columns = [desc[0] for desc in cursor.description]

    return _row_to_dict(result_row, columns)


async def identify_user(name: str, phone: str) -> dict[str, Any]:
    logger.info("Identifying user: name=%s, phone=%s", name, phone)
    return await upsert_user(name, phone)
