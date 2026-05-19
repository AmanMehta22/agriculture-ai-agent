from datetime import datetime
import json
import sqlite3
from pathlib import Path


class Memory:
    def __init__(self):
        self.store_path = Path("reports") / "chat_memory.sqlite3"
        self.messages = []
        self._ensure_store()
        self._load_from_store()

    def _ensure_store(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.store_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _load_from_store(self):
        with sqlite3.connect(self.store_path) as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages ORDER BY id ASC"
            ).fetchall()

        self.messages = [{"role": role, "content": content} for role, content in rows]

    def _append_to_store(self, role, content):
        timestamp = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.store_path) as connection:
            connection.execute(
                "INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)",
                (role, content, timestamp),
            )
            connection.commit()

    def add(self, role, content):
        message = {"role": role, "content": content}
        self.messages.append(message)
        self._append_to_store(role, content)

    def get_all(self):
        return self.messages

    def clear(self):
        self.messages.clear()
        with sqlite3.connect(self.store_path) as connection:
            connection.execute("DELETE FROM messages")
            connection.commit()

    def to_markdown(self):
        lines = ["# Session Memory", ""]
        for message in self.messages:
            role = message.get("role", "unknown").title()
            content = message.get("content", "")
            lines.append(f"## {role}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def save(self, output_dir="reports"):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        markdown_path = path / f"session_memory_{timestamp}.md"
        json_path = path / f"session_memory_{timestamp}.json"

        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        json_path.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False), encoding="utf-8")

        return markdown_path, json_path