import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
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

# Hardcoded slot templates: (day_offset_from_first_business_day, time_slots_24h)
_SLOT_TEMPLATES = [
    (0, ["09:00", "11:00", "14:00"]),
    (1, ["10:00", "13:30", "15:00"]),
    (2, ["09:30", "11:30", "14:30"]),
    (3, ["10:00", "12:00", "15:30"]),
    (4, ["09:00", "11:00", "14:00", "16:00"]),
]


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


def _next_business_days(count: int) -> list:
    """Return the next `count` business days (Mon-Fri) starting from tomorrow."""
    days = []
    current = datetime.now(timezone.utc).date() + timedelta(days=1)
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


async def _seed_slots(db: aiosqlite.Connection) -> None:
    """Insert hardcoded slot data if the table is empty."""
    async with db.execute("SELECT COUNT(*) FROM available_slots") as cursor:
        row = await cursor.fetchone()
        if row and row[0] > 0:
            return

    business_days = _next_business_days(5)
    slot_duration_minutes = 30
    inserts = []
    for day_offset, times in _SLOT_TEMPLATES:
        day_date = business_days[day_offset]
        for time_str in times:
            slot_id = str(uuid.uuid4())
            parts = time_str.split(":")
            start_h, start_m = int(parts[0]), int(parts[1])
            end_total = start_h * 60 + start_m + slot_duration_minutes
            end_h = end_total // 60
            end_m = end_total % 60
            end_time = f"{end_h:02d}:{end_m:02d}"
            inserts.append(
                (
                    slot_id,
                    day_date.isoformat(),
                    time_str,
                    end_time,
                    None,
                    0,
                )
            )

    await db.executemany(
        """
        INSERT INTO available_slots (id, date, start_time, end_time, provider_id, is_booked)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        inserts,
    )
    await db.commit()


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
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS available_slots (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    provider_id TEXT,
                    is_booked INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS appointments (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    slot_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (slot_id) REFERENCES available_slots(id)
                )
                """
            )
            await db.commit()
            await _seed_slots(db)
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


async def get_available_slots(
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Return unbooked slots, optionally filtered by date range."""
    await _ensure_init()

    query = (
        "SELECT id, date, start_time, end_time FROM available_slots WHERE is_booked = 0"
    )
    params: list[str] = []

    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)

    query += " ORDER BY date, start_time"

    async with (
        aiosqlite.connect(DB_PATH) as db,
        db.execute(query, params) as cursor,
    ):
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [_row_to_dict(row, columns) for row in rows]


async def find_slot_by_date_time(date: str, start_time: str) -> dict[str, Any] | None:
    """Find an unbooked slot by date and start_time."""
    await _ensure_init()
    async with (
        aiosqlite.connect(DB_PATH) as db,
        db.execute(
            "SELECT id, date, start_time, end_time FROM available_slots "
            "WHERE date = ? AND start_time = ? AND is_booked = 0",
            (date, start_time),
        ) as cursor,
    ):
        row = await cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return _row_to_dict(row, columns)


async def book_slot(slot_id: str, user_id: str) -> dict[str, Any]:
    """Mark a slot as booked and create an appointment record.

    Raises ValueError if the slot is already booked.
    """
    await _ensure_init()
    async with _upsert_lock:
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT id, is_booked FROM available_slots WHERE id = ?",
                (slot_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise ValueError(f"Slot {slot_id} not found")
            if row[1]:
                raise ValueError(f"Slot {slot_id} is already booked")

            await db.execute(
                "UPDATE available_slots SET is_booked = 1 WHERE id = ?",
                (slot_id,),
            )

            appointment_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO appointments (id, user_id, slot_id, status, created_at, updated_at)
                VALUES (?, ?, ?, 'confirmed', ?, ?)
                """,
                (appointment_id, user_id, slot_id, now, now),
            )
            await db.commit()

            async with db.execute(
                """
                SELECT a.id, a.date, a.start_time, a.end_time
                FROM available_slots a
                WHERE a.id = ?
                """,
                (slot_id,),
            ) as cursor:
                slot_row = await cursor.fetchone()
                slot_columns = [desc[0] for desc in cursor.description]

            return {
                **_row_to_dict(slot_row, slot_columns),
                "appointment_id": appointment_id,
            }


async def get_user_appointments(user_id: str) -> list[dict[str, Any]]:
    """Return confirmed appointments for a user, ordered by date and time."""
    await _ensure_init()

    async with (
        aiosqlite.connect(DB_PATH) as db,
        db.execute(
            """
            SELECT a.id, a.user_id, a.slot_id, s.date, s.start_time, s.end_time, a.status
            FROM appointments a
            JOIN available_slots s ON a.slot_id = s.id
            WHERE a.user_id = ? AND a.status = 'confirmed'
            ORDER BY s.date, s.start_time
            """,
            (user_id,),
        ) as cursor,
    ):
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [_row_to_dict(row, columns) for row in rows]


async def cancel_appointment(appointment_id: str, user_id: str) -> dict[str, Any]:
    """Cancel an appointment and free the associated slot.

    Raises ValueError if the appointment is not found, doesn't belong to the user,
    or is already cancelled.
    """
    await _ensure_init()
    async with _upsert_lock:
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                """
                SELECT a.id, a.user_id, a.slot_id, a.status, s.date, s.start_time, s.end_time
                FROM appointments a
                JOIN available_slots s ON a.slot_id = s.id
                WHERE a.id = ?
                """,
                (appointment_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise ValueError(f"Appointment {appointment_id} not found")

            appt_id, appt_user_id, slot_id, status, date, start_time, end_time = row

            if appt_user_id != user_id:
                raise ValueError(
                    f"Appointment {appointment_id} does not belong to user {user_id}"
                )
            if status == "cancelled":
                raise ValueError(f"Appointment {appointment_id} is already cancelled")

            await db.execute(
                "UPDATE appointments SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (now, appointment_id),
            )
            await db.execute(
                "UPDATE available_slots SET is_booked = 0 WHERE id = ?",
                (slot_id,),
            )
            await db.commit()

            return {
                "id": appt_id,
                "user_id": appt_user_id,
                "slot_id": slot_id,
                "status": "cancelled",
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
            }


async def modify_appointment(
    user_id: str,
    to_be_cancelled_date: str,
    to_be_cancelled_time: str,
    to_be_booked_date: str,
    to_be_booked_time: str,
) -> dict[str, Any]:
    """Modify an appointment by cancelling an existing one and booking a new slot.

    Finds the user's confirmed appointment matching (to_be_cancelled_date, to_be_cancelled_time),
    then books a new slot at (to_be_booked_date, to_be_booked_time).

    Raises ValueError if the old appointment is not found or the new slot is unavailable.
    """
    await _ensure_init()
    async with _upsert_lock:
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                """
                SELECT a.id, a.user_id, a.slot_id, a.status
                FROM appointments a
                JOIN available_slots s ON a.slot_id = s.id
                WHERE a.user_id = ? AND s.date = ? AND s.start_time = ? AND a.status != 'cancelled'
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                (user_id, to_be_cancelled_date, to_be_cancelled_time),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                raise ValueError(
                    f"No appointment found for user {user_id} on {to_be_cancelled_date} at {to_be_cancelled_time}"
                )

            old_appt_id, appt_user_id, old_slot_id, status = row

            if appt_user_id != user_id:
                raise ValueError(f"Appointment does not belong to user {user_id}")
            if status == "cancelled":
                raise ValueError("The appointment is already cancelled")

            async with db.execute(
                "SELECT id, is_booked FROM available_slots WHERE date = ? AND start_time = ?",
                (to_be_booked_date, to_be_booked_time),
            ) as cursor:
                slot_row = await cursor.fetchone()

            if slot_row is None:
                raise ValueError(
                    f"No slot found on {to_be_booked_date} at {to_be_booked_time}"
                )
            if slot_row[1]:
                raise ValueError(
                    f"Slot on {to_be_booked_date} at {to_be_booked_time} is already booked"
                )

            new_slot_id = slot_row[0]

            await db.execute(
                "UPDATE appointments SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (now, old_appt_id),
            )
            await db.execute(
                "UPDATE available_slots SET is_booked = 0 WHERE id = ?",
                (old_slot_id,),
            )
            await db.execute(
                "UPDATE available_slots SET is_booked = 1 WHERE id = ?",
                (new_slot_id,),
            )

            new_appointment_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO appointments (id, user_id, slot_id, status, created_at, updated_at)
                VALUES (?, ?, ?, 'confirmed', ?, ?)
                """,
                (new_appointment_id, user_id, new_slot_id, now, now),
            )
            await db.commit()

            async with db.execute(
                """
                SELECT a.id, a.date, a.start_time, a.end_time
                FROM available_slots a
                WHERE a.id = ?
                """,
                (new_slot_id,),
            ) as cursor:
                slot_result = await cursor.fetchone()
                slot_columns = [desc[0] for desc in cursor.description]

            return {
                **_row_to_dict(slot_result, slot_columns),
                "appointment_id": new_appointment_id,
                "slot_id": new_slot_id,
                "status": "confirmed",
            }
