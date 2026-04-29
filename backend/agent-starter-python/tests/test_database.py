import os
import shutil
import sys
import tempfile

import aiosqlite
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import database


@pytest.fixture(autouse=True)
def temp_db():
    tmpdir = tempfile.mkdtemp()
    db_dir = os.path.join(tmpdir, "test_data")
    old_db_dir = database.DB_DIR
    old_db_path = database.DB_PATH
    database.DB_DIR = db_dir
    database.DB_PATH = os.path.join(db_dir, "test.db")
    database._initialized = False
    yield
    database.DB_DIR = old_db_dir
    database.DB_PATH = old_db_path
    database._initialized = False
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_init_db_creates_table():
    await database._ensure_init()
    assert os.path.exists(database.DB_PATH)

    async with (
        aiosqlite.connect(database.DB_PATH) as db,
        db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ) as cursor,
    ):
        row = await cursor.fetchone()
        assert row is not None


@pytest.mark.asyncio
async def test_upsert_user_creates_new_user():
    user = await database.upsert_user("John Doe", "+1 234 567 8901")
    assert user["name"] == "John Doe"
    assert user["phone"] == "+12345678901"
    assert user["id"] is not None
    assert user["created_at"] is not None
    assert user["updated_at"] is not None


@pytest.mark.asyncio
async def test_upsert_user_updates_name():
    await database.upsert_user("John Doe", "+1 234 567 8902")
    user = await database.upsert_user("John Smith", "+1 234 567 8902")
    assert user["name"] == "John Smith"
    assert user["phone"] == "+12345678902"


@pytest.mark.asyncio
async def test_upsert_user_same_name_no_change():
    await database.upsert_user("John Doe", "+1 234 567 8903")
    user = await database.upsert_user("John Doe", "+1 234 567 8903")
    assert user["name"] == "John Doe"
    assert user["phone"] == "+12345678903"


@pytest.mark.asyncio
async def test_get_user_by_phone_returns_user():
    await database.upsert_user("Jane Doe", "+1 234 567 8904")
    user = await database.get_user_by_phone("+12345678904")
    assert user is not None
    assert user["name"] == "Jane Doe"
    assert user["phone"] == "+12345678904"


@pytest.mark.asyncio
async def test_get_user_by_phone_returns_none():
    user = await database.get_user_by_phone("+12345678999")
    assert user is None


@pytest.mark.asyncio
async def test_phone_uniqueness():
    await database.upsert_user("User A", "+1 234 567 8905")
    await database.upsert_user("User B", "+1 234 567 8906")

    user_a = await database.get_user_by_phone("+12345678905")
    user_b = await database.get_user_by_phone("+12345678906")

    assert user_a["name"] == "User A"
    assert user_b["name"] == "User B"
    assert user_a["id"] != user_b["id"]


@pytest.mark.asyncio
async def test_identify_user_creates():
    user = await database.identify_user("Alice", "+1 234 567 8907")
    assert user["name"] == "Alice"
    assert user["phone"] == "+12345678907"


@pytest.mark.asyncio
async def test_identify_user_updates():
    await database.upsert_user("Bob", "+1 234 567 8908")
    user = await database.identify_user("Robert", "+1 234 567 8908")
    assert user["name"] == "Robert"
    assert user["phone"] == "+12345678908"


# --- Phone validation tests ---


def test_validate_phone_valid_us():
    valid, result = database.validate_phone("+1 234 567 8900")
    assert valid
    assert result == "+12345678900"


def test_validate_phone_valid_india():
    valid, result = database.validate_phone("+91 98765 43210")
    assert valid
    assert result == "+919876543210"


def test_validate_phone_no_plus_prefix():
    valid, result = database.validate_phone("1234567890")
    assert not valid
    assert "+" in result


def test_validate_phone_empty_string():
    valid, result = database.validate_phone("")
    assert not valid
    assert "empty" in result.lower()


def test_validate_phone_whitespace_only():
    valid, result = database.validate_phone("   ")
    assert not valid
    assert "empty" in result.lower()


def test_validate_phone_too_short():
    valid, _ = database.validate_phone("+123")
    assert not valid


def test_validate_phone_non_numeric():
    valid, _ = database.validate_phone("+abc")
    assert not valid


def test_validate_phone_mixed_garbage():
    valid, _ = database.validate_phone("+1 hello 234")
    assert not valid


@pytest.mark.asyncio
async def test_upsert_user_rejects_invalid_phone():
    with pytest.raises(ValueError, match=r"invalid|format|prefix"):
        await database.upsert_user("Bad Phone", "1234567890")


@pytest.mark.asyncio
async def test_upsert_user_rejects_empty_phone():
    with pytest.raises(ValueError, match="empty"):
        await database.upsert_user("Empty Phone", "")


@pytest.mark.asyncio
async def test_upsert_user_normalizes_phone():
    user = await database.upsert_user("Normalize", "+1 234 567 8900")
    assert user["phone"] == "+12345678900"
