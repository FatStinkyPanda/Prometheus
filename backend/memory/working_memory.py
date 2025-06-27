# backend/memory/working_memory.py

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypeVar

import asyncpg

from backend.memory.base_memory import BaseMemory

# Generic TypeVar for the class instance, used for the factory method's return type.
T_WorkingMemory = TypeVar('T_WorkingMemory', bound='WorkingMemory')

class WorkingMemory(BaseMemory):
    """
    Manages the short-term, transient memory of the consciousness system.

    This memory is session-specific and stores key-value pairs with a Time-To-Live (TTL),
    making it suitable for holding temporary context, user preferences for a session,
    or the current focus of attention.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the WorkingMemory instance.

        Args:
            config (Dict[str, Any]): The memory configuration dictionary, expected to contain
                                     a 'working_memory' or 'auto_cleanup' section.
        """
        super().__init__(config)
        # Look for the TTL setting in the config, with a fallback default.
        self.default_ttl = self.config.get("auto_cleanup", {}).get("working_memory_ttl_seconds", 3600)

    @classmethod
    async def create(cls: type[T_WorkingMemory], config: Dict[str, Any]) -> T_WorkingMemory:
        """
        Asynchronous factory method to create and initialize an instance of WorkingMemory.
        """
        instance = cls(config)
        try:
            instance.pool = instance.db_manager.get_pool()
            instance._initialized = True
            instance.logger.info("Successfully initialized and connected to the database pool.")
        except ConnectionError as e:
            instance.logger.critical("Failed to get database pool during WorkingMemory initialization. %s", e, exc_info=True)
            raise
        return instance

    async def add(self, session_id: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> Optional[str]:
        """
        Adds or updates a key-value pair in the working memory for a specific session.
        Uses an "upsert" operation (INSERT ON CONFLICT) to handle both new and existing keys.

        Args:
            session_id (str): The ID of the session.
            key (str): The key for the data.
            value (Any): The Python object to store. Must be JSON-serializable.
            ttl_seconds (Optional[int]): The time-to-live for this entry in seconds.
                                         If None, the default from the config is used.
        Returns:
            Optional[str]: The UUID of the created or updated record, or None on failure.
        """
        self._check_initialized()
        self.logger.debug("Adding/updating key '%s' for session '%s'.", key, session_id)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        
        try:
            value_json = json.dumps(value)
        except TypeError as e:
            self.logger.error("Failed to serialize value for key '%s'. Value must be JSON-serializable. Error: %s", key, e, exc_info=True)
            return None

        query = """
            INSERT INTO working_memory (session_id, key, value, ttl_seconds, last_accessed)
            VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id, key) DO UPDATE
            SET value = EXCLUDED.value,
                ttl_seconds = EXCLUDED.ttl_seconds,
                last_accessed = CURRENT_TIMESTAMP,
                created_at = CURRENT_TIMESTAMP -- Reset TTL on update
            RETURNING id;
        """
        try:
            record_id = await self._fetchval(query, session_id, key, value_json, ttl)
            self.logger.info("Successfully added/updated key '%s' for session '%s'.", key, session_id)
            return str(record_id) if record_id else None
        except asyncpg.PostgresError as e:
            self.logger.error("Database error while adding key '%s' to working memory. Error: %s", key, e, exc_info=True)
            return None

    async def get(self, session_id: str, key: str) -> Optional[Any]:
        """
        Retrieves a value from working memory, if it exists and has not expired.
        This operation also updates the `last_accessed` timestamp for the entry.

        Args:
            session_id (str): The ID of the session.
            key (str): The key of the data to retrieve.
        Returns:
            Optional[Any]: The deserialized Python object, or None if not found or expired.
        """
        self._check_initialized()
        self.logger.debug("Getting key '%s' for session '%s'.", key, session_id)
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                query = """
                    SELECT value FROM working_memory
                    WHERE session_id = $1
                      AND key = $2
                      AND (created_at + (ttl_seconds * INTERVAL '1 second')) > CURRENT_TIMESTAMP;
                """
                value_json = await conn.fetchval(query, session_id, key)

                if value_json is None:
                    self.logger.debug("Key '%s' not found or expired for session '%s'.", key, session_id)
                    return None

                update_query = "UPDATE working_memory SET last_accessed = CURRENT_TIMESTAMP WHERE session_id = $1 AND key = $2"
                await conn.execute(update_query, session_id, key)

        try:
            return json.loads(value_json)
        except json.JSONDecodeError:
            self.logger.error("Failed to deserialize JSON value for key '%s' in session '%s'.", key, session_id, exc_info=True)
            return None

    async def search(self, session_id: str, key_pattern: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Searches for non-expired entries with keys matching a pattern."""
        self._check_initialized()
        self.logger.debug("Searching for keys matching '%s' for session '%s'.", key_pattern, session_id)
        query = """
            SELECT key, value, created_at, last_accessed FROM working_memory
            WHERE session_id = $1
              AND key LIKE $2
              AND (created_at + (ttl_seconds * INTERVAL '1 second')) > CURRENT_TIMESTAMP
            ORDER BY last_accessed DESC
            LIMIT $3;
        """
        records = await self._fetch(query, session_id, key_pattern, limit)
        
        results = []
        for record in records:
            try:
                results.append({
                    "key": record["key"],
                    "value": json.loads(record["value"]),
                    "created_at": record["created_at"],
                    "last_accessed": record["last_accessed"],
                })
            except (json.JSONDecodeError, KeyError):
                self.logger.warning("Could not process record during search: %s", dict(record), exc_info=True)
                continue
        return results

    async def delete(self, session_id: str, key: str) -> bool:
        """Deletes a specific key-value pair from a session's working memory."""
        self._check_initialized()
        self.logger.info("Deleting key '%s' for session '%s'.", key, session_id)
        query = "DELETE FROM working_memory WHERE session_id = $1 AND key = $2;"
        status = await self._execute(query, session_id, key)
        return 'DELETE 1' in status

    async def clear(self, session_id: str) -> None:
        """Clears all working memory entries for a given session."""
        self._check_initialized()
        self.logger.warning("Clearing all working memory for session '%s'.", session_id)
        query = "DELETE FROM working_memory WHERE session_id = $1;"
        await self._execute(query, session_id)

    async def cleanup_expired(self) -> int:
        """Removes all expired entries from the working_memory table across all sessions."""
        self._check_initialized()
        self.logger.info("Running scheduled cleanup of expired working memory entries...")
        query = """
            WITH deleted AS (
                DELETE FROM working_memory
                WHERE (created_at + (ttl_seconds * INTERVAL '1 second')) < CURRENT_TIMESTAMP
                RETURNING id
            )
            SELECT count(*) FROM deleted;
        """
        try:
            deleted_count = await self._fetchval(query)
            if deleted_count > 0:
                self.logger.info("Cleaned up %d expired working memory entries.", deleted_count)
            else:
                self.logger.debug("No expired working memory entries to clean up.")
            return deleted_count
        except Exception as e:
            self.logger.error("An error occurred during working memory cleanup: %s", e, exc_info=True)
            return 0