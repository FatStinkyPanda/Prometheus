# backend/main.py

import sys
import argparse
import asyncio
from typing import Dict, Any, Optional

# --- Pre-emptive check for critical libraries ---
try:
    from backend.utils.logger import Logger
    from backend.utils.config_loader import ConfigLoader
    from backend.database.connection_manager import DatabaseManager
    # Import for the new infinite context integration
    from backend.core.consciousness.unified_consciousness import UnifiedConsciousness
    from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness # Import the wrapper
except ImportError as e:
    print(f"FATAL ERROR: A required local module is missing: {e.name}", file=sys.stderr)
    print("This indicates a problem with the project structure or Python's path.", file=sys.stderr)
    print("Please ensure your virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Try running: python run_prometheus.py --force-setup", file=sys.stderr)
    sys.exit(1)

logger = Logger(__name__)

async def run_headless(config: Dict[str, Any]):
    logger.info("--- Starting Prometheus in Headless (API-only) Mode ---")
    from backend.api.server import PrometheusAPIServer # Import here for headless mode

    main_consciousness_instance: Optional[InfiniteConsciousness] = None
    db_manager = DatabaseManager()

    try:
        # Initialize DatabaseManager if not already done for this event loop
        if not db_manager._initialized or db_manager._pool is None or \
           (db_manager._pool._loop is not asyncio.get_running_loop() if db_manager._pool else True):
            logger.info("Headless: Initializing DatabaseManager for the API event loop.")
            await db_manager.initialize()
        else:
            logger.info("Headless: DatabaseManager already initialized for this event loop.")

        logger.info("Initializing base UnifiedConsciousness engine...")
        base_unified_consciousness = await UnifiedConsciousness.create(config)
        logger.info("Base UnifiedConsciousness engine initialized successfully.")

        logger.info("Wrapping UnifiedConsciousness with InfiniteConsciousness for enhanced capabilities...")
        main_consciousness_instance = InfiniteConsciousness(base_unified_consciousness)
        logger.info("InfiniteConsciousness wrapper initialized. This will be the primary consciousness object for the API.")
        
        # Initialize and run the API server, passing the InfiniteConsciousness instance.
        # PrometheusAPIServer's __init__ and its call to set_consciousness_instance
        # will need to be updated to handle this type.
        api_server = PrometheusAPIServer(main_consciousness_instance, config)
        api_server.run_blocking()

    except Exception as e:
        logger.critical("A fatal error occurred during headless startup: %s", e, exc_info=True)
        if main_consciousness_instance and main_consciousness_instance.consciousness: # Access underlying UC for shutdown
            try:
                await main_consciousness_instance.consciousness.shutdown()
            except Exception as shutdown_err:
                logger.error(f"Error during consciousness shutdown in headless error handler: {shutdown_err}", exc_info=True)
        elif main_consciousness_instance : # Fallback if UC wasn't accessible
             logger.warning("Underlying UnifiedConsciousness not accessible on main_consciousness_instance for shutdown.")
        
        if db_manager._initialized and db_manager._pool and not db_manager._pool.is_closing():
            try:
                await db_manager.close()
            except Exception as db_close_err:
                logger.error(f"Error during DB manager close in headless error handler: {db_close_err}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("--- Prometheus Headless Mode Shutting Down ---")
        if main_consciousness_instance and main_consciousness_instance.consciousness:
            try:
                await main_consciousness_instance.consciousness.shutdown()
            except Exception as shutdown_err:
                logger.error(f"Error during consciousness shutdown in headless finally block: {shutdown_err}", exc_info=True)
        
        if db_manager._initialized and db_manager._pool and not db_manager._pool.is_closing():
            logger.info("Headless: Closing DatabaseManager pool during final shutdown.")
            try:
                await db_manager.close()
            except Exception as db_close_err:
                logger.error(f"Error during DB manager close in headless finally block: {db_close_err}", exc_info=True)

def run_gui(config: Dict[str, Any]):
    logger.info("--- Starting Prometheus in GUI Mode ---")
    try:
        import qasync
        from PyQt6.QtWidgets import QApplication
        from backend.gui.main_window import PrometheusMainWindow
    except ImportError as e:
        logger.critical(f"A required library for GUI mode is missing: {e.name}", exc_info=True)
        print(f"FATAL ERROR: A required library for GUI mode is missing: {e.name}", file=sys.stderr)
        print("Please ensure your virtual environment is activated and dependencies are installed.", file=sys.stderr)
        print("You may need to run the setup script: python run_prometheus.py --force-setup", file=sys.stderr)
        sys.exit(1)

    app: Optional[QApplication] = None
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Prometheus Consciousness System")
        app.setOrganizationName("FatStinkyPanda")
    except Exception as e: # Catch potential errors during QApplication init, e.g. display server issues
        logger.critical("Failed to create the QApplication. This is a fatal error. Exiting. Error: %s", e, exc_info=True)
        sys.exit(1)

    loop: Optional[qasync.QEventLoop] = None
    try:
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        logger.info("qasync event loop initialized and set for the current asyncio context.")
    except Exception as e:
        logger.critical("Failed to create or set the qasync event loop. Error: %s", e, exc_info=True)
        sys.exit(1)


    window: Optional[PrometheusMainWindow] = None
    db_manager_gui = DatabaseManager() 

    # For the GUI, PrometheusMainWindow.initialize_backend creates its own UnifiedConsciousness.
    # If the GUI itself needs to directly use InfiniteConsciousness methods,
    # then PrometheusMainWindow's initialization logic would need to be updated to create
    # or receive an InfiniteConsciousness instance.
    # The API server started *by the GUI* will use the consciousness instance created by MainWindow.
    try:
        window = PrometheusMainWindow() 
        window.show()
        logger.info("PrometheusMainWindow instantiated. Backend (including consciousness) will initialize asynchronously within it.")
    except Exception as e:
        logger.critical("Failed to create or show the PrometheusMainWindow. Error: %s", e, exc_info=True)
        if window: # If window object exists even partially
            try: window.close() # Attempt to trigger its closeEvent for cleanup
            except Exception as e_close: logger.error(f"Error trying to close window during exception handling: {e_close}")

        if db_manager_gui._initialized and db_manager_gui._pool and not db_manager_gui._pool.is_closing():
            logger.info("GUI Mode: Closing DatabaseManager pool due to error during window creation.")
            try:
                if loop and not loop.is_closed():
                    loop.run_until_complete(db_manager_gui.close())
                else: # Fallback if loop is not available
                    asyncio.run(db_manager_gui.close()) # This might create a new loop, less ideal but attempts cleanup
            except Exception as db_close_err:
                logger.error(f"Error closing DB pool during GUI error handling: {db_close_err}")
        sys.exit(1)

    try:
        if loop: # Ensure loop is not None
            with loop: # Context manager for qasync event loop
                logger.info("Handing control to the application event loop...")
                loop.run_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Application is shutting down.")
    except Exception as e: # Catch any other unhandled exceptions from the event loop
        logger.critical("An unhandled exception escaped the event loop. Error: %s", e, exc_info=True)
    finally:
        if window and window.isVisible(): 
            logger.info("GUI Mode: Closing main window to trigger cleanup...")
            window.close() # This should trigger PrometheusMainWindow.closeEvent for its specific cleanup
        
        if loop and not loop.is_closed():
            loop.close()
        logger.info("Event loop terminated. Application exiting.")

def main(args: argparse.Namespace):
    logger.info("--- Starting Prometheus Consciousness System v3.0 ---")
    try:
        config = ConfigLoader.load_config() # Uses defaults from ConfigLoader
    except Exception as e:
        logger.critical("Failed to load configuration. This is a fatal error. Exiting. Error: %s", e, exc_info=True)
        sys.exit(1)

    if args.headless or args.api_only:
        if sys.platform == "win32" and not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(run_headless(config))
    else:
        run_gui(config)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Run the Prometheus Consciousness System.",
            formatter_class=argparse.RawTextHelpFormatter # Preserves formatting in help text
        )
        parser.add_argument(
            "--headless",
            action="store_true",
            help="Run the backend without a GUI, serving only the API."
        )
        parser.add_argument(
            "--api-only",
            action="store_true",
            help="Alias for --headless."
        )
        # Consider adding a --config-file argument here in the future if needed
        parsed_args = parser.parse_args()
        main(parsed_args)
    except SystemExit: # Raised by argparse's --help or our own sys.exit calls.
        pass # Allow clean exit
    except Exception as e: # Catch-all for truly unexpected errors at the very top level
        # This is a last resort if logging isn't even set up or fails.
        error_message = f"FATAL: An unhandled exception occurred at the top level: {e}"
        print(error_message, file=sys.stderr) # Print to stderr as logger might not be up
        try: # Try to log it if logger was set up
            logger.critical(error_message, exc_info=True)
        except NameError: # Logger not defined (e.g., if ConfigLoader itself failed critically)
            pass 
        except Exception: # Logger itself failed
            pass 
        sys.exit(1) # Exit with error code