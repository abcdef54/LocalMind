"""Conversation HTTP integration tests against an explicitly disposable PostgreSQL DB."""

import asyncio
from datetime import datetime, timezone
import os
import unittest
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.backend.api.application import install_api_foundation
from src.backend.api.v1.dependencies import get_session
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Attachment, Conversation, Message, Run, Submission

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class ConversationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        url = make_url(TEST_URL)
        if url.database != "localmind_migration_tests":
            raise RuntimeError("Tests require disposable database localmind_migration_tests")
        cls.engine = sa.create_engine(url.set(drivername="postgresql+psycopg2"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.dispose()

    def setUp(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))
            upgrade_database(connection)
        self.async_engine = create_async_engine(TEST_URL, poolclass=NullPool)
        self.sessions = async_sessionmaker(self.async_engine, expire_on_commit=False)
        self.queries = []

        @sa.event.listens_for(self.async_engine.sync_engine, "before_cursor_execute")
        def capture_queries(connection, cursor, statement, parameters, context, executemany):
            self.queries.append((statement, parameters))

        async def session_dependency():
            async with self.sessions() as session:
                yield session

        app = FastAPI()
        versioned_api = install_api_foundation(app)
        versioned_api.dependency_overrides[get_session] = session_dependency
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.addCleanup(lambda: asyncio.run(self.async_engine.dispose()))

    def seed_conversation(self, *, conversation_id: UUID | None = None) -> UUID:
        conversation_id = conversation_id or uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Conversation).values(
                    id=conversation_id,
                    title="Existing conversation",
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            )
        return conversation_id

    def seed_messages(self, conversation_id: UUID, count: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Message),
                [
                    {
                        "id": UUID(int=index),
                        "conversation_id": conversation_id,
                        "content": f"message {index}",
                        "role": "user",
                        "created_at": datetime(2026, 1, 1),
                    }
                    for index in range(1, count + 1)
                ],
            )

    def history_url(self, conversation_id: UUID) -> str:
        return f"/api/v1/conversations/{conversation_id}/messages"

    def test_create_read_rename_and_archive_conversation(self) -> None:
        response = self.client.post("/api/v1/conversations", json={"title": "  Research  "})
        self.assertEqual(response.status_code, 201, response.text)
        conversation_id = response.json()["id"]
        url = f"/api/v1/conversations/{conversation_id}"
        self.assertEqual(response.json()["title"], "Research")
        self.assertEqual(self.client.get(url).status_code, 200)
        response = self.client.patch(url, json={"title": "Renamed"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["title"], "Renamed")
        response = self.client.delete(url)
        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(response.content, b"")
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.delete(url).status_code, 404)
        self.assertEqual(self.client.get(self.history_url(UUID(conversation_id))).status_code, 404)
        self.assertEqual(self.client.get("/api/v1/conversations").json()["items"], [])

    def test_large_history_pages_use_sql_limit_and_stable_timestamp_tiebreaker(self) -> None:
        conversation_id = self.seed_conversation()
        self.seed_messages(conversation_id, 1005)
        response = self.client.get(self.history_url(conversation_id))
        self.assertEqual(response.status_code, 200, response.text)
        page = response.json()
        self.assertEqual(
            [UUID(message["id"]).int for message in page["items"]], list(range(956, 1006))
        )
        self.assertTrue(all(message["created_at"].endswith("Z") for message in page["items"]))
        seen_ids = []
        page_count = 0
        while True:
            page_count += 1
            page_ids = [UUID(message["id"]).int for message in page["items"]]
            self.assertEqual(page_ids, sorted(page_ids))
            self.assertLessEqual(len(page_ids), 50)
            seen_ids.extend(page_ids)
            if not page["has_more"]:
                self.assertIsNone(page["next_cursor"])
                break
            page = self.client.get(
                self.history_url(conversation_id), params={"before": page["next_cursor"]}
            ).json()
        self.assertEqual(page_count, 21)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))
        self.assertEqual(set(seen_ids), set(range(1, 1006)))
        history_queries = [
            (statement, parameters)
            for statement, parameters in self.queries
            if statement.startswith("SELECT messages.")
        ]
        self.assertEqual(len(history_queries), 21)
        for statement, parameters in history_queries:
            self.assertIn("LIMIT", statement)
            self.assertEqual(parameters[-1], 51)
            self.assertNotIn("OFFSET", statement)
        self.assertFalse(any("count(" in statement.lower() for statement, _ in self.queries))

    def test_cursor_survives_new_insert_and_deleted_anchor(self) -> None:
        conversation_id = self.seed_conversation()
        self.seed_messages(conversation_id, 120)
        first = self.client.get(self.history_url(conversation_id)).json()
        with self.engine.begin() as connection:
            connection.execute(sa.delete(Message).where(Message.id == UUID(int=71)))
            connection.execute(
                sa.insert(Message).values(
                    id=UUID(int=121),
                    conversation_id=conversation_id,
                    content="new",
                    role="user",
                    created_at=datetime(2026, 1, 2),
                )
            )
        response = self.client.get(
            self.history_url(conversation_id), params={"before": first["next_cursor"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [UUID(message["id"]).int for message in response.json()["items"]], list(range(21, 71))
        )

    def test_invalid_cursor_scope_limits_and_missing_conversations(self) -> None:
        conversation_id = self.seed_conversation()
        other_id = self.seed_conversation()
        self.seed_messages(conversation_id, 51)
        cursor = self.client.get(self.history_url(conversation_id)).json()["next_cursor"]
        response = self.client.get(self.history_url(other_id), params={"before": cursor})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_CURSOR")
        for token in ("bad-cursor", "e30", "☃"):
            with self.subTest(token=token):
                response = self.client.get(
                    self.history_url(conversation_id), params={"before": token}
                )
                self.assertEqual(response.status_code, 422)
        for limit in (0, -1, 101, "bad"):
            with self.subTest(limit=limit):
                self.assertEqual(
                    self.client.get(
                        self.history_url(conversation_id), params={"limit": limit}
                    ).status_code,
                    422,
                )
        self.assertEqual(self.client.get(self.history_url(uuid4())).status_code, 404)
        self.assertEqual(self.client.get("/api/v1/conversations/not-a-uuid").status_code, 422)

    def test_empty_history_and_conversation_title_validation(self) -> None:
        conversation_id = self.seed_conversation()
        self.assertEqual(
            self.client.get(self.history_url(conversation_id)).json(),
            {"items": [], "next_cursor": None, "has_more": False},
        )
        for payload in (
            {"title": " "},
            {"title": "x" * 256},
            {"title": "ok", "user_id": "old"},
            {},
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/api/v1/conversations", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_conversation_list_pagination_is_scoped_separately(self) -> None:
        for index in range(1, 8):
            self.seed_conversation(conversation_id=UUID(int=index))
        first = self.client.get("/api/v1/conversations", params={"limit": 3}).json()
        self.assertEqual([UUID(row["id"]).int for row in first["items"]], [7, 6, 5])
        second = self.client.get(
            "/api/v1/conversations", params={"limit": 3, "before": first["next_cursor"]}
        ).json()
        self.assertEqual([UUID(row["id"]).int for row in second["items"]], [4, 3, 2])
        self.assertEqual(
            self.client.get(
                self.history_url(UUID(int=1)), params={"before": first["next_cursor"]}
            ).status_code,
            422,
        )

    def test_history_filters_deleted_messages_and_cross_conversation_citations(self) -> None:
        conversation_id = self.seed_conversation()
        other_id = self.seed_conversation()
        self.seed_messages(conversation_id, 3)
        own_file_id, foreign_file_id = uuid4(), uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Message)
                .where(Message.id == UUID(int=2))
                .values(deleted_at=datetime(2026, 1, 2))
            )
            for file_id, owner_id in ((own_file_id, conversation_id), (foreign_file_id, other_id)):
                connection.execute(
                    sa.insert(Attachment).values(
                        id=file_id,
                        conversation_id=owner_id,
                        file_name="source.pdf",
                        file_path="/private/path.pdf",
                        file_type="application/pdf",
                        file_size=12,
                        deleted_at=datetime(2026, 1, 2) if owner_id == conversation_id else None,
                    )
                )
                connection.execute(
                    sa.insert(
                        sa.table(
                            "message_attachments",
                            sa.column("message_id"),
                            sa.column("attachment_id"),
                        )
                    ).values(message_id=UUID(int=1), attachment_id=file_id)
                )
        response = self.client.get(self.history_url(conversation_id))
        self.assertEqual(response.status_code, 200, response.text)
        messages = response.json()["items"]
        self.assertEqual([UUID(row["id"]).int for row in messages], [1, 3])
        self.assertEqual([row["id"] for row in messages[0]["citations"]], [str(own_file_id)])
        self.assertTrue(messages[0]["citations"][0]["deleted"])
        self.assertNotIn("private/path", response.text)

    def test_active_run_blocks_conversation_deletion(self) -> None:
        conversation_id = self.seed_conversation()
        submission_id, message_id = uuid4(), uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Submission).values(
                    id=submission_id,
                    conversation_id=conversation_id,
                    client_submission_id=uuid4(),
                    fingerprint="a" * 64,
                    settings_snapshot={"model_name": "local"},
                )
            )
            connection.execute(
                sa.insert(Message).values(
                    id=message_id,
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    role="user",
                    content="question",
                )
            )
            connection.execute(
                sa.insert(Run).values(
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    message_id=message_id,
                    settings_snapshot={"model_name": "local"},
                )
            )
        response = self.client.delete(f"/api/v1/conversations/{conversation_id}")
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"]["code"], "RUN_ACTIVE")
        self.assertEqual(
            self.client.get(f"/api/v1/conversations/{conversation_id}").status_code, 200
        )

    def test_archive_hides_owned_metadata_but_preserves_other_conversations(self) -> None:
        conversation_id = self.seed_conversation()
        other_id = self.seed_conversation()
        self.seed_messages(conversation_id, 2)
        attachment_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Attachment).values(
                    id=attachment_id,
                    conversation_id=conversation_id,
                    file_name="kept-for-cleanup.pdf",
                    file_path="/not-read-during-metadata-archive.pdf",
                    file_type="application/pdf",
                    file_size=5,
                )
            )
        response = self.client.delete(f"/api/v1/conversations/{conversation_id}")
        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(self.client.get(f"/api/v1/conversations/{other_id}").status_code, 200)
        with self.engine.connect() as connection:
            self.assertIsNotNone(
                connection.scalar(
                    sa.select(Attachment.deleted_at).where(Attachment.id == attachment_id)
                )
            )
            self.assertEqual(
                connection.scalar(
                    sa.select(sa.func.count())
                    .select_from(Message)
                    .where(Message.deleted_at.is_(None))
                ),
                0,
            )
            self.assertEqual(
                connection.scalar(sa.select(sa.func.count()).select_from(Attachment)), 1
            )


if __name__ == "__main__":
    unittest.main()
