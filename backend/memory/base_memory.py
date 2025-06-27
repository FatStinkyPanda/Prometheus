# backend/memory/base_memory.py

import abc
import asyncpg
from typing import Dict, Any, List, Optional, TypeVar

from backend.database.connection_manager import DatabaseManager
from backend.utils.logger import Logger

# Generic TypeVar for the class instance, used for the factory method's return type.
T = TypeVar('T', bound='BaseMemory')

class BaseMemory(abc.ABC):
    """
    An abstract base class for all memory systems in Prometheus.

    This class provides a common structure and interface for memory operations,
    ensuring that all memory subclasses have a consistent design. It handles
    the acquisition of the database connection pool and provides helper methods
    for executing queries with robust error handling.

    Subclasses must implement the abstract methods defined here.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the BaseMemory instance.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration for this memory system.
        """
        self.config = config
        self.logger = Logger(self.__class__.__name__)  # Logger named after the subclass
        self.db_manager = DatabaseManager()
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    @classmethod
    async def create(cls: type[T], config: Dict[str, Any]) -> T:
        """
        An asynchronous factory method to create and initialize an instance of a memory class.

        Args:
            config (Dict[str, Any]): Configuration specific to the memory system.

        Returns:
            An initialized instance of the memory subclass.
        """
        instance = cls(config)
        try:
            instance.pool = instance.db_manager.get_pool()
            instance._initialized = True
            instance.logger.info("Successfully initialized and connected to the database pool.")
        except ConnectionError as e:
            instance.logger.critical("Failed to get database pool during initialization. %s", e, exc_info=True)
            # Re-raise to ensure the application fails fast if the DB isn't ready.
            raise
        return instance

    def _check_initialized(self):
        """Raises a RuntimeError if the memory system is not initialized."""
        if not self._initialized or not self.pool:
            raise RuntimeError(
                f"{self.__class__.__name__} is not initialized. "
                "Ensure `await .create()` was called."
            )

    async def _execute(self, query: str, *params) -> str:
        """
        Executes a command that doesn't return rows (INSERT, UPDATE, DELETE).

        Args:
            query (str): The SQL query to execute.
            params: Parameters to pass to the query.

        Returns:
            str: The status message from the database command (e.g., 'INSERT 1').
        """
        self._check_initialized()
        try:
            async with self.pool.acquire() as conn:
                return await conn.execute(query, *params)
        except asyncpg.PostgresError as e:
            self.logger.error("Database command failed. Query: %s, Error: %s", query, e, exc_info=True)
            raise  # Re-raise the exception to be handled by the caller.

    async def _fetch(self, query: str, *params) -> List[asyncpg.Record]:
        """
        Executes a query that returns multiple rows.

        Args:
            query (str): The SQL query to execute.
            params: Parameters to pass to the query.

        Returns:
            List[asyncpg.Record]: A list of row records.
        """
        self._check_initialized()
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetch(query, *params)
        except asyncpg.PostgresError as e:
            self.logger.error("Database fetch failed. Query: %s, Error: %s", query, e, exc_info=True)
            raise

    async def _fetchrow(self, query: str, *params) -> Optional[asyncpg.Record]:
        """
        Executes a query that is expected to return at most one row.

        Args:
            query (str): The SQL query to execute.
            params: Parameters to pass to the query.

        Returns:
            Optional[asyncpg.Record]: A single row record, or None if no row was found.
        """
        self._check_initialized()
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchrow(query, *params)
        except asyncpg.PostgresError as e:
            self.logger.error("Database fetchrow failed. Query: %s, Error: %s", query, e, exc_info=True)
            raise

    async def _fetchval(self, query: str, *params) -> Any:
        """
        Executes a query that returns a single value from a single row.

        Args:
            query (str): The SQL query to execute.
            params: Parameters to pass to the query.

        Returns:
            Any: The single value, or None if no row was found.
        """
        self._check_initialized()
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, *params)
        except asyncpg.PostgresError as e:
            self.logger.error("Database fetchval failed. Query: %s, Error: %s", query, e, exc_info=True)
            raise

    # --- Abstract methods to be implemented by all subclasses ---

    @abc.abstractmethod
    async def add(self, *args, **kwargs) -> Any:
        """Add an entry to this memory system."""
        pass

    @abc.abstractmethod
    async def get(self, *args, **kwargs) -> Optional[Any]:
        """Get a specific entry from this memory system."""
        pass

    @abc.abstractmethod
    async def search(self, *args, **kwargs) -> List[Any]:
        """Search for entries in this memory system."""
        pass

    @abc.abstractmethod
    async def delete(self, *args, **kwargs) -> bool:
        """Delete an entry from this memory system."""
        pass

    @abc.abstractmethod
    async def clear(self, *args, **kwargs) -> None:
        """Clear all entries from this memory system, optionally filtered."""
        pass