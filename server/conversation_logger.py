"""
Conversation logging system for the chatbot with SQLite storage.
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Optional
import os
from loguru import logger

class ConversationLogger:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database with the conversations table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        speaker TEXT NOT NULL,
                        message_text TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)
                """)
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def generate_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())

    def log_conversation(self, session_id: str, speaker: str, message_text: str, timestamp: Optional[datetime] = None):
        """
        Log a conversation message to the database.
        
        Args:
            session_id: Unique identifier for the conversation session
            speaker: Either 'user' or 'bot'
            message_text: The actual conversation text
            timestamp: When the message occurred (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO conversations (session_id, timestamp, speaker, message_text)
                    VALUES (?, ?, ?, ?)
                """, (session_id, timestamp, speaker, message_text))
                conn.commit()
                logger.debug(f"Logged {speaker} message for session {session_id}: {message_text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> list:
        """
        Retrieve conversation history for a given session.
        
        Args:
            session_id: The session ID to retrieve
            limit: Maximum number of messages to return (most recent first)
            
        Returns:
            List of tuples (timestamp, speaker, message_text)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if limit:
                    cursor = conn.execute("""
                        SELECT timestamp, speaker, message_text 
                        FROM conversations 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT timestamp, speaker, message_text 
                        FROM conversations 
                        WHERE session_id = ? 
                        ORDER BY timestamp ASC
                    """, (session_id,))
                
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []

    def get_all_sessions(self) -> list:
        """
        Get all unique session IDs from the database.
        
        Returns:
            List of session IDs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT session_id FROM conversations ORDER BY created_at DESC")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve sessions: {e}")
            return []

    def delete_session(self, session_id: str):
        """Delete all messages for a given session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
                conn.commit()
                logger.info(f"Deleted session {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")