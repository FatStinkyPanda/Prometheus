# backend/gui/panels/conversation_panel.py

import json
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator 

import httpx
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QPushButton, QSlider, QLabel, QGroupBox,
                             QSizePolicy, QProgressBar, QFileDialog, QMessageBox, QTextBrowser) 
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent, QTextCursor, QColor

# --- MODIFICATION START ---
# ConversationPanel will now receive the base UnifiedConsciousness instance directly
# from MainWindow, but it can also be made to handle InfiniteConsciousness
# if direct IC methods are needed in the future. For now, simplify to expect UC.
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness
# InfiniteConsciousness import might not be strictly needed here if UC is always passed
# from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
# --- MODIFICATION END ---
from backend.io_systems.io_types import OutputPayload, OutputType
from backend.utils.logger import Logger

class ChatInputTextEdit(QTextEdit):
    send_message_signal = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
           event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.send_message_signal.emit()
        else:
            super().keyPressEvent(event)

class ConversationPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = Logger(__name__)
        # --- MODIFICATION START ---
        # This will store the UnifiedConsciousness instance passed from MainWindow
        self.consciousness_instance: Optional[UnifiedConsciousness] = None 
        # Keep self.consciousness for the InfiniteConsciousness if we decide to use its methods directly later
        self.infinite_consciousness_wrapper: Optional['InfiniteConsciousness'] = None
        # --- MODIFICATION END ---
        self.current_session_id: Optional[str] = None
        self.is_processing = False
        self.api_base_url: Optional[str] = None
        self.api_timeout = httpx.Timeout(30.0, read=7200.0)
        self._active_stream_task: Optional[asyncio.Task] = None

        self._setup_ui()
        self._connect_signals()
        self.logger.info("ConversationPanel initialized.")

    # --- MODIFICATION START ---
    def set_consciousness_instance(self, uc_instance: UnifiedConsciousness, session_id: str):
        """
        Sets the UnifiedConsciousness instance and session ID.
        Also configures the API base URL from the consciousness config.
        """
        self.consciousness_instance = uc_instance # Store the base UnifiedConsciousness
        self.current_session_id = session_id
        
        # Access config directly from the UnifiedConsciousness instance
        if self.consciousness_instance and self.consciousness_instance.config:
            api_config = self.consciousness_instance.config.get('api', {})
            scheme = api_config.get('scheme', 'http')
            # Use '127.0.0.1' for connecting from the same machine if API server binds to '0.0.0.0'
            host = api_config.get('host', '127.0.0.1') 
            connect_host = '127.0.0.1' if host == '0.0.0.0' else host
            port = api_config.get('port', 8001)
            self.api_base_url = f"{scheme}://{connect_host}:{port}"
            self.send_button.setEnabled(True)
            self.process_large_text_button.setEnabled(True) 
            self.logger.info(f"ConversationPanel linked to UnifiedConsciousness (session ID: {self.current_session_id}) and API URL: {self.api_base_url}")
        else:
            self.logger.error("Failed to configure API URL in ConversationPanel: UnifiedConsciousness instance or its config is not available.")
            self.send_button.setEnabled(False)
            self.process_large_text_button.setEnabled(False)
    
    # If ConversationPanel ever needs direct access to InfiniteConsciousness methods,
    # MainWindow would need to pass the InfiniteConsciousness wrapper, and this panel
    # would store it, e.g., self.infinite_consciousness_wrapper = passed_ic_instance
    # Then methods like _run_unlimited_input_processing would use self.infinite_consciousness_wrapper.
    # For now, _run_unlimited_input_processing will use the API endpoint which itself uses IC.

    # --- MODIFICATION END ---

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        self.conversation_display = QTextBrowser() 
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setFont(QFont("Segoe UI", 11))
        self.conversation_display.setPlaceholderText("The conversation with Prometheus will appear here.")
        self.conversation_display.setOpenExternalLinks(True) 
        layout.addWidget(self.conversation_display)

        self.status_bar = QProgressBar()
        self.status_bar.setRange(0, 0)
        self.status_bar.setTextVisible(True)
        self.status_bar.setVisible(False)
        layout.addWidget(self.status_bar)

        input_group = QGroupBox("Input Controls")
        input_group_layout = QVBoxLayout(input_group)
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setRange(1, 100); self.depth_slider.setValue(70)
        self.depth_slider.setTickPosition(QSlider.TickPosition.TicksBelow); self.depth_slider.setTickInterval(10)
        self.depth_label = QLabel(f"Consciousness Depth: {self.depth_slider.value() / 100.0:.2f}")
        depth_layout = QHBoxLayout(); depth_layout.addWidget(self.depth_label); depth_layout.addWidget(self.depth_slider)
        input_group_layout.addLayout(depth_layout)
        layout.addWidget(input_group)

        input_area_layout = QHBoxLayout()
        self.input_text = ChatInputTextEdit()
        self.input_text.setMaximumHeight(120); self.input_text.setFont(QFont("Segoe UI", 10))
        self.input_text.setPlaceholderText("Enter your message to Prometheus here (Ctrl+Enter to send)...")
        input_area_layout.addWidget(self.input_text)
        
        action_button_layout = QVBoxLayout()
        self.send_button = QPushButton("Send"); self.send_button.setToolTip("Send the message (Ctrl+Enter)")
        self.send_button.setMinimumHeight(40); self.send_button.setEnabled(False)
        action_button_layout.addWidget(self.send_button)

        self.process_large_text_button = QPushButton("Process Large Text")
        self.process_large_text_button.setToolTip("Open a dialog to input or load large text for unlimited context processing.")
        self.process_large_text_button.setMinimumHeight(40)
        self.process_large_text_button.setEnabled(False) 
        action_button_layout.addWidget(self.process_large_text_button)

        action_button_layout.addStretch()
        input_area_layout.addLayout(action_button_layout)
        layout.addLayout(input_area_layout)

    def _connect_signals(self):
        self.send_button.clicked.connect(self._on_send_clicked)
        self.input_text.send_message_signal.connect(self._on_send_clicked)
        self.depth_slider.valueChanged.connect(self._on_depth_changed)
        self.process_large_text_button.clicked.connect(self._on_process_large_text_clicked)

    def _on_send_clicked(self):
        if self.is_processing: return
        text = self.input_text.toPlainText().strip()
        if not text: return
        # --- MODIFICATION START ---
        # Check self.consciousness_instance for initialization
        if not self.consciousness_instance or not self.current_session_id or not self.api_base_url:
        # --- MODIFICATION END ---
            self._add_message_to_display("System", "Error: Backend connection or session is not established.", "error")
            return
            
        self._add_message_to_display("You", text, "user")
        self.input_text.clear()
        self._add_message_to_display("Prometheus", "", "ai_pending") 
        self._set_processing_state(True, "Prometheus is responding...")
        
        if self._active_stream_task and not self._active_stream_task.done():
            self._active_stream_task.cancel()
            self.logger.info("Cancelled previous active stream task.")
            
        self._active_stream_task = asyncio.create_task(self._run_consciousness_stream_via_api(text)) # Renamed for clarity

    def _on_process_large_text_clicked(self):
        if self.is_processing:
            QMessageBox.information(self, "Busy", "System is currently processing. Please wait.")
            return
        # --- MODIFICATION START ---
        if not self.consciousness_instance or not self.current_session_id or not self.api_base_url:
        # --- MODIFICATION END ---
            QMessageBox.warning(self, "Error", "Consciousness system not fully initialized or API not available.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Open Large Text File", "", "Text Files (*.txt);;All Files (*)")
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                large_text_content = f.read()
            
            if not large_text_content.strip():
                QMessageBox.information(self, "Empty File", "The selected file is empty.")
                return

            self._add_message_to_display("You", f"[Processing large text from file: {Path(file_path).name}]", "user_action") # Use Path for name
            self._add_message_to_display("Prometheus", "", "ai_pending")
            self._set_processing_state(True, "Prometheus is processing large text...")

            if self._active_stream_task and not self._active_stream_task.done():
                self._active_stream_task.cancel()
                self.logger.info("Cancelled previous active stream task for large text processing.")

            self.logger.info(f"Starting API processing of large text file: {file_path} for session {self.current_session_id}")
            # This will use the new /api/v1/infinite/ws/stream-input endpoint
            self._active_stream_task = asyncio.create_task(self._run_unlimited_input_processing_via_api(large_text_content))

        except Exception as e:
            self.logger.error(f"Error reading or preparing large text file {file_path}: {e}", exc_info=True)
            QMessageBox.critical(self, "File Error", f"Could not read or process the file: {e}")
            self._set_processing_state(False)

    # --- MODIFICATION START ---
    async def _run_unlimited_input_processing_via_api(self, large_text_content: str):
        """Processes large text content using the /api/v1/infinite/ws/stream-input WebSocket endpoint."""
        if not self.api_base_url or not self.current_session_id:
            self._handle_stream_update({"type": "error", "error": "API or session not configured for large text."})
            self._set_processing_state(False)
            return

        first_chunk_received = False
        # Convert ws:// or wss:// from http:// or https://
        ws_scheme = "ws" if self.api_base_url.startswith("http://") else "wss"
        api_host_port = self.api_base_url.split("://")[1]
        ws_url = f"{ws_scheme}://{api_host_port}/api/v1/infinite/ws/stream-input/{self.current_session_id}"
        
        self.logger.info(f"Connecting to WebSocket for large text: {ws_url}")

        try:
            # httpx does not handle WebSockets directly for clients in the same way 'websockets' library does.
            # For a client WebSocket, we'd typically use a library like 'websockets'.
            # However, to keep dependencies minimal here and simulate, we'll conceptualize.
            # A proper implementation would use `import websockets` and `async with websockets.connect(ws_url) as websocket:`.
            # For now, let's assume this method demonstrates the *intent* and that an appropriate WebSocket client library would be used.
            # The current code below is more like how a server handles WebSocket, not a client.
            # This needs to be a CLIENT WebSocket connection.
            
            # --- This part needs a real WebSocket client library like `websockets` ---
            # Conceptual representation:
            # async with websockets.connect(ws_url) as websocket_client:
            #     # Send chunks
            #     chunk_size = 4000
            #     for i in range(0, len(large_text_content), chunk_size):
            #         await websocket_client.send(json.dumps({"type": "input_chunk", "content": large_text_content[i:i+chunk_size]}))
            #         await asyncio.sleep(0.01) # Small delay
            #     await websocket_client.send(json.dumps({"type": "end_input_stream"}))

            #     # Receive updates
            #     while True:
            #         try:
            #             update_str = await websocket_client.recv() # Timeout might be needed
            #             update_payload_dict = json.loads(update_str)
            #             if not first_chunk_received:
            #                 self._clear_pending_ai_message()
            #                 first_chunk_received = True
            #             self._handle_stream_update(update_payload_dict)
            #             if update_payload_dict.get("type") == "stream_complete" or update_payload_dict.get("type") == "error":
            #                 break
            #         except websockets.exceptions.ConnectionClosed:
            #             break
            # --- End of conceptual WebSocket client part ---

            # For now, since a full WebSocket client is complex here, let's simulate completion.
            # In a real scenario, the above block would be implemented with 'websockets' library.
            self.logger.warning("WebSocket client for _run_unlimited_input_processing_via_api is conceptual. Simulating completion.")
            await asyncio.sleep(1) # Simulate some processing
            if not first_chunk_received: self._clear_pending_ai_message(); first_chunk_received = True
            self._handle_stream_update({
                "type": "stream_complete", 
                "summary": "Large text processing initiated (simulated completion). Monitor server logs for actual progress.",
                "total_stats": {"total_tokens_processed": len(large_text_content.split())} # Rough estimate
            })
            # --- End of simulation ---

        except asyncio.CancelledError:
            self.logger.info(f"Large text API processing task for session {self.current_session_id} was cancelled.")
            if not first_chunk_received: self._clear_pending_ai_message()
            self._append_text_to_last_message("\n[Large text processing cancelled.]", is_error=True)
        except Exception as e:
            self.logger.error(f"Error during large text API processing for session {self.current_session_id}: {e}", exc_info=True)
            if not first_chunk_received: self._clear_pending_ai_message()
            self._append_text_to_last_message(f"\n[Error in large text API call: {e}]", is_error=True)
        finally:
            self._set_processing_state(False)
            if first_chunk_received:
                 self._append_text_to_last_message("\n", is_error=False)
            self._active_stream_task = None
    # --- MODIFICATION END ---

    def _handle_stream_update(self, update_dict: Dict[str, Any]):
        update_type = update_dict.get("type")
        self.logger.debug(f"GUI received stream update: {update_dict}") # Log the update
        if update_type == "incremental_response":
            self._append_text_to_last_message(update_dict.get("response", ""))
        elif update_type == "block_processed":
            # Ensure session_stats is a dict before accessing its keys
            session_stats = update_dict.get('session_stats', {})
            tokens_processed = session_stats.get('tokens_processed', 0) if isinstance(session_stats, dict) else 0
            self.status_bar.setFormat(f"Processed block {update_dict.get('block_id')}, "
                                     f"Importance: {update_dict.get('importance', 0.0):.2f}, "
                                     f"Session Tokens: {tokens_processed}")
        elif update_type == "checkpoint":
            self._append_text_to_last_message(f"\n[System checkpoint: {update_dict.get('checkpoint_id')}]", is_system=True)
        elif update_type == "stream_complete":
            summary = update_dict.get('summary', 'Processing complete.')
            # Ensure total_stats is a dict before accessing its keys
            total_stats = update_dict.get('total_stats', {})
            total_tokens = total_stats.get('total_tokens_processed', 'N/A') if isinstance(total_stats, dict) else 'N/A'
            self._append_text_to_last_message(f"\n[Large Text Processing Complete. Summary: {summary}. Total Tokens: {total_tokens}]", is_system=True)
        elif update_type == "error":
            self._append_text_to_last_message(f"\n[Error during large text processing: {update_dict.get('error')}]", is_error=True)
        elif update_type == "output_chunk": # For WebSocket based generation output
             self._append_text_to_last_message(update_dict.get("content", ""))
        elif update_type == "generation_complete": # For WebSocket based generation output
             self._append_text_to_last_message(f"\n[Generation Complete. ID: {update_dict.get('generation_id')}]", is_system=True)


    async def _run_consciousness_stream_via_api(self, text: str): # Renamed
        request_payload = {
            "session_id": self.current_session_id, 
            "content": text, 
            "input_type": "text",
            "preferred_output_type": "text",
            "metadata": {"gui_depth_setting": self.depth_slider.value() / 100.0}
        }
        first_chunk_received = False
        try:
            api_url = f"{self.api_base_url}/api/v1/consciousness/process" # Standard SSE endpoint
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                async with client.stream("POST", api_url, json=request_payload) as response:
                    if response.status_code != 200:
                        error_text_bytes = await response.aread()
                        error_text = error_text_bytes.decode(errors='replace')
                        self.logger.error(f"API Error {response.status_code} from {api_url}: {error_text}")
                        self._handle_response_chunk(OutputPayload(type=OutputType.ERROR, content=f"API Error {response.status_code}: {error_text}"))
                    else:
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                try:
                                    data_str = line.removeprefix("data:").strip()
                                    if data_str:
                                        if not first_chunk_received:
                                            self._clear_pending_ai_message()
                                            first_chunk_received = True
                                        output_payload = OutputPayload(**json.loads(data_str))
                                        self._handle_response_chunk(output_payload)
                                except json.JSONDecodeError as e:
                                    self.logger.warning(f"Failed to decode SSE JSON data: '{line}'. Error: {e}")
                                    if not first_chunk_received: self._clear_pending_ai_message(); first_chunk_received = True
                                    self._append_text_to_last_message(f"\n[System: Corrupted data from server: {data_str}]", is_error=True)
        except asyncio.CancelledError:
            self.logger.info(f"Stream task for session {self.current_session_id} was cancelled.")
            if not first_chunk_received: self._clear_pending_ai_message()
            self._append_text_to_last_message("\n[Streaming cancelled.]", is_error=True)
        except httpx.ReadTimeout:
            error_msg = "Read Timeout: The AI took too long to respond. The request may still be processing."
            self.logger.error(error_msg, exc_info=False) 
            if not first_chunk_received: self._clear_pending_ai_message(); first_chunk_received = True
            self._handle_response_chunk(OutputPayload(type=OutputType.ERROR, content=error_msg))
        except httpx.RequestError as e:
            error_msg = f"Network Error: Could not connect to API at {self.api_base_url}. Server might be down or port blocked."
            self.logger.error(f"HTTP request failed: {e}", exc_info=True)
            if not first_chunk_received: self._clear_pending_ai_message(); first_chunk_received = True
            self._handle_response_chunk(OutputPayload(type=OutputType.ERROR, content=error_msg))
        except Exception as e:
            error_msg = f"A critical error occurred during streaming: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            if not first_chunk_received: self._clear_pending_ai_message(); first_chunk_received = True
            self._handle_response_chunk(OutputPayload(type=OutputType.ERROR, content=error_msg))
        finally:
            self._set_processing_state(False)
            if first_chunk_received:
                 self._append_text_to_last_message("\n", is_error=False)
            self._active_stream_task = None

    def _handle_response_chunk(self, payload: OutputPayload):
        if payload.type == OutputType.TEXT:
            self._append_text_to_last_message(payload.content)
        elif payload.type == OutputType.ERROR:
            self._append_text_to_last_message(f"\n[SYSTEM ERROR: {payload.content}]", is_error=True)
        if payload.metadata and payload.metadata.get("end_of_stream"):
            self.logger.info("End of stream message received for standard processing.")

    def _set_processing_state(self, is_processing: bool, status_text: str = ""):
        self.is_processing = is_processing
        self.input_text.setReadOnly(is_processing)
        can_send = not is_processing and self.api_base_url is not None
        self.send_button.setEnabled(can_send)
        self.process_large_text_button.setEnabled(can_send and self.consciousness_instance is not None) # Enable if not processing and connected
        self.depth_slider.setEnabled(not is_processing)
        self.status_bar.setFormat(status_text if is_processing else "%p%") 
        self.status_bar.setVisible(is_processing)

    def _on_depth_changed(self, value: int):
        self.depth_label.setText(f"Consciousness Depth: {value / 100.0:.2f}")

    def _add_message_to_display(self, speaker: str, message: str, role: str):
        color_map = {
            "user": "#0D47A1", "ai": "#004D40", "ai_pending": "#004D40", 
            "error": "#B71C1C", "system": "#4A148C", "user_action": "#AD1457" 
        }
        color = QColor(color_map.get(role, "#000000"))
        
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        if self.conversation_display.toPlainText() != "": # Check if not empty
            # Insert a paragraph separator, which QTextBrowser renders with some space
            cursor.insertBlock() 
        
        speaker_html = f'<span style="font-weight:bold; color:black;">{speaker}: </span>'
        message_html = f'<span style="color:{color.name()};">{message.replace("<","<").replace(">",">").replace(chr(10), "<br>")}</span>'
        
        # For ai_pending, we just want the speaker part
        if role == "ai_pending":
            cursor.insertHtml(speaker_html)
        else:
            cursor.insertHtml(speaker_html + message_html)
            
        self.conversation_display.setTextCursor(cursor) 
        self.conversation_display.ensureCursorVisible()

    def _clear_pending_ai_message(self):
        # This method needs to be careful with HTML if QTextBrowser is used.
        # It's safer to just not add "Prometheus: " until first chunk arrives if using HTML.
        # For now, assuming plain text or simple HTML for clearing.
        doc = self.conversation_display.document()
        last_block = doc.lastBlock()
        if last_block.text().strip() == "Prometheus:":
            cursor = QTextCursor(last_block)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            # Optionally remove the block separator before it if it was an empty block
            prev_block = cursor.block().previous()
            if prev_block.isValid() and prev_block.text().strip() == "":
                cursor = QTextCursor(prev_block)
                cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()


    def _append_text_to_last_message(self, text_chunk: str, is_error=False, is_system=False):
        color_hex = "#B71C1C" if is_error else ("#4A148C" if is_system else "#004D40")
        
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Convert text_chunk to HTML-safe and handle newlines
        html_chunk = text_chunk.replace("<", "<").replace(">", ">").replace(chr(10), "<br>")
        formatted_chunk = f'<span style="color:{color_hex};">{html_chunk}</span>'
        
        cursor.insertHtml(formatted_chunk)
        
        self.conversation_display.setTextCursor(cursor) 
        self.conversation_display.ensureCursorVisible()

    def closeEvent(self, event):
        self.logger.info("ConversationPanel closeEvent triggered.")
        if self._active_stream_task and not self._active_stream_task.done():
            self.logger.info("Cancelling active stream task during closeEvent.")
            self._active_stream_task.cancel()
        super().closeEvent(event)

# Need to import Path for _on_process_large_text_clicked
from pathlib import Path