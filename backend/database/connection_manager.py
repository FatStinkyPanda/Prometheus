# backend/database/connection_manager.py

import asyncpg
import asyncio
import threading
from typing import Dict, Any, Optional

from backend.utils.config_loader import ConfigLoader
from backend.utils.logger import Logger

class DatabaseManager:
    """
    A Singleton class to manage the asyncpg connection pool for the application.

    This manager ensures that there is only one connection pool instance created,
    which is essential for managing database resources efficiently in an async application.
    It handles initialization, provides connections, and manages graceful shutdown.
    """
    _instance: Optional['DatabaseManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                # Double-check locking to ensure thread safety
                if not cls._instance:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization if the instance already exists
        if hasattr(self, '_pool'):
            return

        self.logger = Logger(__name__)
        self._pool: Optional[asyncpg.Pool] = None
        self._config: Dict[str, Any] = self._load_db_config()
        self._initialized = False

    def _load_db_config(self) -> Dict[str, Any]:
        """Loads database configuration using the ConfigLoader."""
        try:
            full_config = ConfigLoader.load_config(
                primary_config="prometheus_config.yaml"
            )
            return full_config.get('database', {})
        except Exception as e:
            self.logger.error("Failed to load database configuration: %s", e, exc_info=True)
            # Return an empty dict to prevent crashes, initialization will fail later
            return {}

    async def initialize(self):
        """
        Initializes the database connection pool.

        This method should be called once during application startup.
        It is idempotent and will not re-initialize if already called.
        """
        if self._initialized:
            self.logger.info("DatabaseManager is already initialized.")
            return

        if not self._config:
            self.logger.critical("Database configuration is missing. Cannot initialize pool.")
            raise ConnectionError("Database configuration is missing.")

        self.logger.info("Initializing database connection pool...")
        try:
            self._pool = await asyncpg.create_pool(
                user=self._config.get('user'),
                password=self._config.get('password'),
                database=self._config.get('name'),
                host=self._config.get('host'),
                port=self._config.get('port'),
                min_size=self._config.get('pool_min_size', 5),
                max_size=self._config.get('pool_max_size', 20),
                timeout=self._config.get('pool_acquire_timeout', 30)
            )
            await self._test_connection()
            self._initialized = True
            self.logger.info("Database connection pool initialized successfully to %s:%s on db '%s'.",
                             self._config.get('host'), self._config.get('port'), self._config.get('name'))
        except (asyncpg.exceptions.PostgresError, ConnectionRefusedError, OSError) as e:
            self.logger.critical("Failed to initialize database connection pool: %s", e, exc_info=True)
            self._pool = None
            raise ConnectionError(f"Could not connect to the database: {e}")
        except Exception as e:
            self.logger.critical("An unexpected error occurred during database initialization: %s", e, exc_info=True)
            self._pool = None
            raise

    async def _test_connection(self):
        """Tests the connection and verifies required extensions."""
        if not self._pool:
            raise ConnectionError("Connection pool is not available for testing.")

        async with self._pool.acquire() as conn:
            # Test basic connection and get PostgreSQL version
            version = await conn.fetchval("SELECT version();")
            self.logger.info("Database connection test successful. PostgreSQL version: %s", version)

            # Test for vector extension, which is critical
            # Note: The extension name is 'vector', not 'pgvector'
            vector_exists = await conn.fetchval(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            if vector_exists:
                self.logger.info("Verified 'vector' extension is enabled in the database.")
                
                # Test vector functionality
                try:
                    await conn.fetchval("SELECT '[1,2,3]'::vector")
                    self.logger.info("Vector extension is functioning correctly.")
                except Exception as e:
                    self.logger.critical("Vector extension is installed but not functioning: %s", e)
                    raise RuntimeError("Vector extension test failed.")
            else:
                self.logger.critical("The 'vector' extension is not enabled in the database. "
                                     "Please run 'CREATE EXTENSION IF NOT EXISTS \"vector\";' in your database.")
                raise RuntimeError("Required 'vector' database extension not found.")

    def get_pool(self) -> asyncpg.Pool:
        """
        Returns the active connection pool.

        Returns:
            asyncpg.Pool: The active connection pool.

        Raises:
            ConnectionError: If the pool has not been initialized.
        """
        if not self._initialized or not self._pool:
            self.logger.error("Attempted to get database pool before it was initialized.")
            raise ConnectionError("DatabaseManager is not initialized. Call initialize() first.")
        return self._pool

    async def close(self):
        """Gracefully closes the database connection pool."""
        if self._pool and not self._pool.is_closing():
            self.logger.info("Closing database connection pool...")
            try:
                await self._pool.close()
                self.logger.info("Database connection pool closed successfully.")
            except Exception as e:
                self.logger.error("Error while closing database connection pool: %s", e, exc_info=True)
        self._pool = None
        self._initialized = False

async def main():
    """Self-test function for the DatabaseManager."""
    logger = Logger(__name__)
    logger.info("--- Running DatabaseManager Self-Test ---")
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        
        pool = db_manager.get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                result = await connection.fetchval("SELECT 2 + 2;")
                logger.info("Test query 'SELECT 2 + 2;' returned: %s", result)
                assert result == 4, "Test query failed!"
        logger.info("Self-test successful.")
    except Exception as e:
        logger.error("Self-test FAILED: %s", e, exc_info=True)
    finally:
        await db_manager.close()
        logger.info("--- Self-Test Complete ---")

if __name__ == '__main__':
    # To run this test, ensure you have a PostgreSQL server running with the
    # credentials specified in your config files and the vector extension enabled.
    asyncio.run(main())