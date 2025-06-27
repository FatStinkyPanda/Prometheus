# backend/api/server.py

import uvicorn
import threading
import asyncio 
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# --- MODIFICATION START ---
# Import InfiniteConsciousness for type hinting the main consciousness object
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
# UnifiedConsciousness might still be needed if we need to access the underlying base instance directly
# from backend.core.consciousness.unified_consciousness import UnifiedConsciousness
# --- MODIFICATION END ---

from backend.utils.logger import Logger

# Import the route modules
from backend.api.routes import (
    consciousness as consciousness_routes,
    conversation as conversation_routes,
    memory as memory_routes,
    system as system_routes,
    streaming as streaming_routes 
)
from backend.api.infinite_context_endpoints import setup_infinite_context_api
from backend.api.dependencies import set_consciousness_instance


class PrometheusAPIServer:
    """
    Manages the FastAPI application for the Prometheus Consciousness System.
    """

    # --- MODIFICATION START ---
    # Update constructor to expect InfiniteConsciousness
    def __init__(self, consciousness: InfiniteConsciousness, config: Dict[str, Any]):
    # --- MODIFICATION END ---
        self.logger = Logger(__name__)
        self.consciousness = consciousness # This is now an InfiniteConsciousness instance
        self.config = config
        self.api_config = config.get('api', {})
        self.system_config = config.get('system', {})
        
        self.app = FastAPI(
            title="Prometheus Consciousness API",
            version=self.system_config.get('version', '3.0.0'),
            description="API for interacting with the Prometheus Consciousness System.",
            lifespan=self._lifespan,
            docs_url="/docs",
            redoc_url="/redoc",
        )
        
        self._configure_middleware()
        self._configure_routes() 

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self.logger.info("--- Prometheus API Server Starting Up ---")
        # --- MODIFICATION START ---
        # self.consciousness is now InfiniteConsciousness, pass it to set_consciousness_instance
        set_consciousness_instance(self.consciousness)
        self.logger.info("Global InfiniteConsciousness instance has been set for all API routes.")
        # --- MODIFICATION END ---

        try:
            self.logger.info("Setting up Infinite Context API endpoints...")
            # setup_infinite_context_api expects the consciousness object.
            # We modified it to handle either UnifiedConsciousness or InfiniteConsciousness.
            # Since self.consciousness is now InfiniteConsciousness, this will work.
            await setup_infinite_context_api(self.app, self.consciousness)
            self.logger.info("Infinite Context API endpoints configured.")
        except Exception as e:
            self.logger.critical(f"Failed to setup Infinite Context API: {e}", exc_info=True)
        
        yield 
        
        self.logger.info("--- Prometheus API Server Shutting Down ---")


    def _configure_middleware(self):
        self.logger.info("Configuring API middleware...")
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.api_config.get('cors_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.logger.info(f"CORS configured for origins: {self.api_config.get('cors_origins', ['*'])}")
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            self.logger.error(
                f"Unhandled exception during API request to '{request.url.path}': {exc}",
                exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={"detail": f"An internal server error occurred: {str(exc)}"}
            )
        self.logger.info("Global exception handler configured.")

    def _configure_routes(self):
        self.logger.info("Configuring API routes...")
        
        # These routes will now use get_consciousness() which returns InfiniteConsciousness
        self.app.include_router(consciousness_routes.router, prefix="/api/v1/consciousness", tags=["Consciousness"])
        self.app.include_router(conversation_routes.router, prefix="/api/v1/conversation", tags=["Conversation"])
        self.app.include_router(memory_routes.router, prefix="/api/v1/memory", tags=["Memory"])
        self.app.include_router(system_routes.router, prefix="/api/v1/system", tags=["System"])
        self.app.include_router(streaming_routes.router, prefix="/api/v1/streaming", tags=["Streaming (General)"])
        self.logger.info("Included general streaming routes under /api/v1/streaming.")
        
        @self.app.get("/", tags=["System"])
        async def root():
            return {
                "name": "Prometheus Consciousness API",
                "version": self.app.version,
                "docs_url": self.app.docs_url,
                "redoc_url": self.app.redoc_url,
                "health_url": "/health"
            }

        @self.app.get("/health", tags=["System"])
        async def health_check():
            current_state = "UNKNOWN"
            # self.consciousness is InfiniteConsciousness, so we access its .consciousness (UC) attribute for state
            if self.consciousness and self.consciousness.consciousness and \
               hasattr(self.consciousness.consciousness, 'state') and self.consciousness.consciousness.state:
                current_state = self.consciousness.consciousness.state.name
            else:
                self.logger.warning("Health check: Underlying UnifiedConsciousness or its state is not available.")

            return {
                "status": "healthy",
                "consciousness_state": current_state
            }
        
        self.logger.info("All API routers have been included.")

    def _get_uvicorn_log_config(self) -> Dict[str, Any]:
        log_config_from_file = self.config.get('logging') if isinstance(self.config, dict) else {}
        if not isinstance(log_config_from_file, dict):
            log_config_from_file = {} 

        log_level = log_config_from_file.get('level', 'INFO').upper()
        
        default_fmt = "%(levelprefix)s %(asctime)s - %(message)s"
        default_datefmt = "%Y-%m-%d %H:%M:%S"
        access_fmt = '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": log_config_from_file.get('format', default_fmt), 
                    "datefmt": log_config_from_file.get('datefmt', default_datefmt), 
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": access_fmt, 
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": log_level, "propagate": False},
                "uvicorn.error": {"level": log_level, "handlers": ["default"], "propagate": False}, 
                "uvicorn.access": {"handlers": ["access"], "level": log_level, "propagate": False},
            },
        }

    def run_in_thread(self) -> threading.Thread:
        host = self.api_config.get('host', '0.0.0.0')
        port = self.api_config.get('port', 8001) 
        
        def run_server():
            try:
                uvicorn.run(
                    self.app,
                    host=host,
                    port=port,
                    log_config=self._get_uvicorn_log_config(),
                )
            except Exception as e:
                self.logger.critical(f"Uvicorn server thread failed: {e}", exc_info=True)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        self.logger.info(f"Uvicorn server started in a background thread. Listening on http://{host}:{port}")
        return server_thread

    def run_blocking(self):
        host = self.api_config.get('host', '0.0.0.0')
        port = self.api_config.get('port', 8001) 
        self.logger.info(f"Starting Uvicorn server in blocking mode at http://{host}:{port}")
        
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_config=self._get_uvicorn_log_config(),
            )
        except Exception as e:
            self.logger.critical(f"Failed to start Uvicorn server: {e}", exc_info=True)
            raise