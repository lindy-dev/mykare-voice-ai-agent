import os
import shutil
import sys
import tempfile

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
async def test_identify_user_tool_creates_new_user():

    from agent import Assistant

    assistant = Assistant()
    result = await assistant.identify_user(
        context=None, name="Test User", phone="+1 234 567 8910"
    )
    assert "Test User" in result
    assert "+12345678910" in result
    assert "user ID" in result


@pytest.mark.asyncio
async def test_identify_user_tool_returns_existing():
    from agent import Assistant

    await database.upsert_user("Existing User", "+1 234 567 8911")
    assistant = Assistant()
    result = await assistant.identify_user(
        context=None, name="Existing User", phone="+1 234 567 8911"
    )
    assert "Existing User" in result
    assert "+12345678911" in result


@pytest.mark.asyncio
async def test_identify_user_tool_updates_name():
    from agent import Assistant

    await database.upsert_user("Old Name", "+1 234 567 8912")
    assistant = Assistant()
    result = await assistant.identify_user(
        context=None, name="New Name", phone="+1 234 567 8912"
    )
    assert "New Name" in result
    assert "+12345678912" in result


@pytest.mark.asyncio
async def test_identify_user_multiple_calls_same_phone():
    from agent import Assistant

    assistant = Assistant()
    r1 = await assistant.identify_user(
        context=None, name="First Call", phone="+1 234 567 8913"
    )
    r2 = await assistant.identify_user(
        context=None, name="Second Call", phone="+1 234 567 8913"
    )
    assert "Second Call" in r2
    assert "+12345678913" in r1
    assert "+12345678913" in r2
