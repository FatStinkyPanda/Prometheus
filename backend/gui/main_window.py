# backend/gui/main_window.py

import sys
import asyncio
import threading
import httpx # For API health check
from typing import Dict, Any, Optional
import uuid

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QTabWidget, QStatusBar, QMessageBox,
                             QDockWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from backend.utils.logger import Logger
from backend.utils.config_loader import ConfigLoader
# --- MODIFICATION START ---
# Import both UnifiedConsciousness and InfiniteConsciousness
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness, ConsciousnessState
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
# --- MODIFICATION END ---
from backend.hardware.resource_manager import HardwareResourceManager
from backend.gui.panels.consciousness_panel import ConsciousnessPanel
from backend.gui.panels.conversation_panel import ConversationPanel
from backend.gui.panels.minds_panel import MindsPanel
from backend.gui.panels.memory_panel import MemoryPanel
from backend.gui.panels.system_panel import SystemPanel
from backend.gui.widgets.thought_stream import ThoughtStream
from backend.api.server import PrometheusAPIServer
from backend.database.connection_manager import DatabaseManager

class PrometheusMainWindow(QMainWindow):
    backend_initialized = pyqtSignal(bool, str) 
    new_thought_event = pyqtSignal(dict) 
    consciousness_cycle_completed = pyqtSignal(dict) 

    def __init__(self):
        super().__init__()
        self.logger = Logger(__name__)
        
        try:
            self.config = ConfigLoader.load_config()
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to load configuration files: {e}\nApplication will exit.")
            sys.exit(1)

        self.resource_manager = HardwareResourceManager()
        self.resource_manager.start_monitoring()

        # --- MODIFICATION START ---
        # self.consciousness will now be an InfiniteConsciousness instance
        self.consciousness: Optional[InfiniteConsciousness] = None
        # --- MODIFICATION END ---
        self.api_server_thread: Optional[threading.Thread] = None
        self.current_session_id = f"gui_session_{uuid.uuid4().hex[:12]}" 
        
        self.state_update_timer = QTimer(self)

        self._setup_ui()
        self._connect_signals()

        self.logger.info("Prometheus Main Window Initialized. Backend initialization will start shortly.")
        self.status_bar.showMessage("Initializing backend consciousness engine...")
        QTimer.singleShot(0, lambda: asyncio.create_task(self.initialize_backend()))

    def _setup_ui(self):
        self.setWindowTitle("Prometheus Consciousness System v3.0")
        self.setGeometry(100, 100, 1800, 1200) 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        self.conversation_panel = ConversationPanel()
        main_splitter.addWidget(self.conversation_panel)

        right_panel_tabs = QTabWidget()
        main_splitter.addWidget(right_panel_tabs)
        main_splitter.setSizes([1000, 800]) 

        self.consciousness_panel = ConsciousnessPanel()
        right_panel_tabs.addTab(self.consciousness_panel, "Consciousness")
        self.minds_panel = MindsPanel()
        right_panel_tabs.addTab(self.minds_panel, "Minds")
        self.memory_panel = MemoryPanel()
        right_panel_tabs.addTab(self.memory_panel, "Memory")
        self.system_panel = SystemPanel()
        right_panel_tabs.addTab(self.system_panel, "System")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        thought_dock_widget = QDockWidget("Autonomous Event Stream", self)
        self.thought_stream = ThoughtStream(thought_dock_widget)
        thought_dock_widget.setWidget(self.thought_stream)
        thought_dock_widget.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, thought_dock_widget)

    def _connect_signals(self):
        self.backend_initialized.connect(self._on_backend_initialized)
        self.consciousness_panel.start_thinking_requested.connect(self._start_thinking)
        self.consciousness_panel.start_dreaming_requested.connect(self._start_dreaming)
        self.consciousness_panel.stop_autonomous_requested.connect(self._stop_autonomous)
        self.new_thought_event.connect(self.thought_stream.add_event)
        self.consciousness_cycle_completed.connect(self.minds_panel.update_displays)
        self.state_update_timer.timeout.connect(self._update_gui_state)

    def _on_new_thought_event(self, event_data: Dict[str, Any]):
        self.new_thought_event.emit(event_data)
        if event_data.get('type') in ['maintenance', 'maintenance_error']:
            self.consciousness_panel.add_log_entry(event_data['content'], event_data['type'])

    async def initialize_backend(self):
        try:
            db_manager = DatabaseManager()
            if not db_manager._initialized or db_manager._pool is None or \
               (db_manager._pool._loop is not asyncio.get_running_loop() if db_manager._pool else True):
                self.logger.info("MainWindow: Initializing DatabaseManager for the GUI event loop.")
                await db_manager.initialize()
            else:
                self.logger.info("MainWindow: DatabaseManager already initialized for this event loop.")

            # --- MODIFICATION START ---
            self.logger.info("MainWindow: Initializing base UnifiedConsciousness engine...")
            base_unified_consciousness = await UnifiedConsciousness.create(self.config, thought_callback=self._on_new_thought_event)
            self.logger.info("MainWindow: Base UnifiedConsciousness engine initialized.")

            self.logger.info("MainWindow: Wrapping UnifiedConsciousness with InfiniteConsciousness...")
            self.consciousness = InfiniteConsciousness(base_unified_consciousness) # self.consciousness is now InfiniteConsciousness
            self.logger.info("MainWindow: InfiniteConsciousness wrapper initialized.")
            # --- MODIFICATION END ---

            self.backend_initialized.emit(True, "")
        except Exception as e:
            self.logger.critical("Failed to initialize the backend (consciousness engine).", exc_info=True)
            self.backend_initialized.emit(False, str(e))

    def _on_backend_initialized(self, success: bool, error_message: str):
        if success and self.consciousness:
            self.logger.info("Backend successfully initialized. Linking to GUI panels and starting API server.")
            self.status_bar.showMessage("Backend initialized. Attempting to start API server...")
            
            try:
                # --- MODIFICATION START ---
                # Pass the InfiniteConsciousness instance to PrometheusAPIServer
                api_server = PrometheusAPIServer(self.consciousness, self.config)
                # --- MODIFICATION END ---
                self.api_server_thread = api_server.run_in_thread()
                QTimer.singleShot(1000, lambda: asyncio.create_task(self._check_api_server_health()))
            except Exception as e:
                self.logger.error(f"Failed to start the API server thread: {e}", exc_info=True)
                QMessageBox.warning(self, "API Server Failed", f"The background API server could not be started.\n\nError: {e}")
                self._link_panels_to_consciousness() 
                self.state_update_timer.start(1000)
        else:
            self.status_bar.showMessage(f"FATAL ERROR: Backend initialization failed. {error_message}")
            QMessageBox.critical(self, "Backend Initialization Failed", f"Could not initialize the consciousness engine.\n\nError: {error_message}")

    async def _check_api_server_health(self):
        if not self.consciousness or not self.consciousness.consciousness: # Check underlying UC
            self.logger.error("Cannot check API health: Consciousness (underlying UC) not available.")
            return
        
        api_config = self.consciousness.consciousness.config.get('api', {}) # Access config via underlying UC
        host = api_config.get('host', '0.0.0.0')
        connect_host = '127.0.0.1' if host == '0.0.0.0' else host
        port = api_config.get('port', 8001)
        health_url = f"http://{connect_host}:{port}/health"
        
        self.logger.info(f"Performing health check on API server at {health_url}...")
        self.status_bar.showMessage("Verifying API server status...")

        for attempt in range(5):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(health_url, timeout=1.0)
                    response.raise_for_status() 
                    if response.json().get("status") == "healthy":
                        self.logger.info("API server health check successful. System is fully operational.")
                        self.status_bar.showMessage("Prometheus is online and ready.", 5000)
                        self._link_panels_to_consciousness() 
                        self.state_update_timer.start(1000) 
                        return
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                self.logger.warning(f"API health check attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1) 
        
        self.logger.critical("API server failed to start or respond to health checks.")
        self.status_bar.showMessage("FATAL ERROR: API server failed to start.")
        QMessageBox.critical(self, "API Server Error", 
            f"Could not connect to the backend API server at {health_url}.\n\n"
            "Please check the console output for errors. The port might be in use, "
            "or a firewall might be blocking the connection.")
        self._link_panels_to_consciousness(api_ready=False)
        self.state_update_timer.start(1000)

    def _link_panels_to_consciousness(self, api_ready: bool = True):
        if not self.consciousness or not self.consciousness.consciousness: # Check underlying UC
            self.logger.error("Cannot link panels: Consciousness (underlying UC) instance is not available.")
            return

        # Panels that need the base UnifiedConsciousness instance
        base_uc_instance = self.consciousness.consciousness
        self.consciousness_panel.set_consciousness_instance(base_uc_instance)
        self.memory_panel.set_consciousness_instance(base_uc_instance) 
        self.system_panel.set_resource_manager(self.resource_manager)
        
        if api_ready:
            # ConversationPanel might need the base UC or the wrapped IC depending on its implementation
            # For now, assuming it uses the API and thus doesn't directly care about the instance type here,
            # but needs a valid consciousness config source.
            self.conversation_panel.set_consciousness_instance(base_uc_instance, self.current_session_id)
        else:
            self.conversation_panel.send_button.setEnabled(False)
            self.conversation_panel.input_text.setPlaceholderText("API Server connection failed. Conversation disabled.")
            self.logger.warning("API Server not ready. ConversationPanel functionality will be limited.")

    def _update_gui_state(self):
        if self.consciousness and self.consciousness.consciousness: # Check underlying UC
            self.consciousness_panel.update_state(self.consciousness.consciousness.state) # Use UC state

    def _start_thinking(self):
        if not self.consciousness or not self.consciousness.consciousness: return # Check underlying UC
        self.consciousness_panel.set_autonomous_mode(True, 'Thinking')
        # Autonomous actions are on UnifiedConsciousness
        asyncio.create_task(self._run_autonomous_action(self.consciousness.consciousness.start_thinking))

    def _start_dreaming(self):
        if not self.consciousness or not self.consciousness.consciousness: return # Check underlying UC
        self.consciousness_panel.set_autonomous_mode(True, 'Dreaming')
        # Autonomous actions are on UnifiedConsciousness
        asyncio.create_task(self._run_autonomous_action(self.consciousness.consciousness.start_dreaming))

    def _stop_autonomous(self):
        if not self.consciousness or not self.consciousness.consciousness: return # Check underlying UC
        # Autonomous actions are on UnifiedConsciousness
        asyncio.create_task(self._run_autonomous_action(self.consciousness.consciousness.stop_autonomous_processing, is_stop=True))

    async def _run_autonomous_action(self, action_coro, is_stop=False):
        try:
            await action_coro()
            if is_stop:
                self.consciousness_panel.set_autonomous_mode(False)
        except Exception as e:
            self.logger.error(f"Error during autonomous action: {e}", exc_info=True)
            self.consciousness_panel.set_autonomous_mode(False)
            QMessageBox.warning(self, "Autonomous Action Error", f"An error occurred: {e}")

    async def _run_consciousness_processing(self, data: Dict[str, Any]):
        # This method is for direct GUI-to-Consciousness processing if ever needed.
        # Most interactions now go through the API via ConversationPanel.
        if not self.consciousness or not self.consciousness.consciousness: # Check underlying UC
            error_payload = {"status": "error", "output": "Critical Error: Consciousness Engine is offline."}
            self.consciousness_cycle_completed.emit(error_payload)
            return error_payload
            
        try:
            # Interactions like these use the base UnifiedConsciousness's process_input
            response_payload = await self.consciousness.consciousness.process_input(
                input_text=data.get("text"), 
                session_id=data.get("session_id")
            )
            self.consciousness_cycle_completed.emit(response_payload)
            return response_payload
        except Exception as e:
            self.logger.error("An unhandled exception occurred during consciousness processing.", exc_info=True)
            error_payload = {"status": "error", "output": f"A critical error occurred: {e}"}
            self.consciousness_cycle_completed.emit(error_payload)
            return error_payload

    def closeEvent(self, event):
        self.logger.info("Close event triggered. Shutting down Prometheus gracefully...")
        self.status_bar.showMessage("Shutting down...")
        
        self.state_update_timer.stop()
        self.resource_manager.stop_monitoring()
        
        if self.api_server_thread and self.api_server_thread.is_alive():
            self.logger.info("Signaling API server shutdown (Uvicorn in thread relies on daemon nature or process exit).")

        if self.consciousness and self.consciousness.consciousness: # Check underlying UC
            try:
                current_loop = asyncio.get_event_loop()
                if current_loop.is_running():
                    # Schedule shutdown of the underlying UnifiedConsciousness
                    shutdown_task = current_loop.create_task(
                        asyncio.wait_for(self.consciousness.consciousness.shutdown(), timeout=10.0)
                    )
                    self.logger.info("Scheduled underlying UnifiedConsciousness shutdown on existing event loop.")
                else:
                    self.logger.info("No event loop running, attempting synchronous shutdown call for UC (less ideal).")
                    asyncio.run(asyncio.wait_for(self.consciousness.consciousness.shutdown(), timeout=10.0))
            except asyncio.TimeoutError:
                self.logger.warning("Underlying UnifiedConsciousness shutdown timed out.")
            except RuntimeError as e: 
                self.logger.warning(f"RuntimeError during UC shutdown (likely event loop already closed): {e}")
            except Exception as e:
                self.logger.error(f"Error during UC shutdown: {e}", exc_info=True)

        self.logger.info("Prometheus GUI shutdown sequence initiated. Accepting close event.")
        event.accept()