# backend/gui/panels/memory_panel.py

import asyncio
import json
from typing import Dict, Any, Optional, List

import httpx
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel,
                             QPushButton, QHBoxLayout, QLineEdit, QComboBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QTextEdit, QAbstractItemView, QSlider, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from backend.utils.logger import Logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.core.consciousness.unified_consciousness import UnifiedConsciousness

class MemoryPanel(QWidget):
    """
    A GUI panel for searching and inspecting the contents of the system's
    memory stores, now using API calls to interact with the backend.
    """
    api_url_set = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = Logger(__name__)
        self.consciousness: Optional['UnifiedConsciousness'] = None
        self._current_search_task: Optional[asyncio.Task] = None
        self.api_base_url: Optional[str] = None
        
        self._setup_ui()
        self._connect_signals()
        self.search_button.setEnabled(False)
        self.logger.info("MemoryPanel initialized.")

    def set_consciousness_instance(self, consciousness: 'UnifiedConsciousness'):
        self.consciousness = consciousness
        api_config = consciousness.config.get('api', {})
        scheme = api_config.get('scheme', 'http')
        host = api_config.get('host', '127.0.0.1')
        connect_host = '127.0.0.1' if host == '0.0.0.0' else host
        port = api_config.get('port', 8001)
        self.api_base_url = f"{scheme}://{connect_host}:{port}"
        self.search_button.setEnabled(True)
        self.logger.info(f"MemoryPanel linked to API at {self.api_base_url}")

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        search_group = QGroupBox("Memory Search")
        search_layout = QVBoxLayout(search_group)
        query_layout = QHBoxLayout()
        self.memory_type_combo = QComboBox()
        self.memory_type_combo.addItems(["Hierarchical", "Truth", "Dream"])
        self.memory_type_combo.setToolTip("Select the memory system to search.")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter semantic search query (or leave blank for history)...")
        self.search_button = QPushButton("Search")
        query_layout.addWidget(QLabel("Memory Type:"))
        query_layout.addWidget(self.memory_type_combo)
        query_layout.addWidget(self.search_input, 1)
        query_layout.addWidget(self.search_button)
        search_layout.addLayout(query_layout)
        
        self.hierarchical_filter_frame = QFrame()
        self.hierarchical_filter_frame.setFrameShape(QFrame.Shape.StyledPanel)
        h_filter_layout = QVBoxLayout(self.hierarchical_filter_frame)
        h_filter_layout.setContentsMargins(5, 5, 5, 5)
        filter_layout = QHBoxLayout()
        self.session_id_input = QLineEdit()
        self.session_id_input.setPlaceholderText("Optional: Filter by Session ID")
        self.importance_slider = QSlider(Qt.Orientation.Horizontal)
        self.importance_slider.setRange(0, 100)
        self.importance_slider.setValue(0)
        self.importance_slider.setToolTip("Filter memories by minimum importance score.")
        self.importance_label = QLabel("Min Importance: 0.00")
        filter_layout.addWidget(QLabel("Session ID:"))
        filter_layout.addWidget(self.session_id_input)
        filter_layout.addSpacing(20)
        filter_layout.addWidget(self.importance_label)
        filter_layout.addWidget(self.importance_slider, 1)
        h_filter_layout.addLayout(filter_layout)
        search_layout.addWidget(self.hierarchical_filter_frame)
        main_layout.addWidget(search_group)

        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Score/Sim", "Type/Tier", "Content Snippet", "Timestamp", "Importance"])
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(self.results_table)
        main_layout.addWidget(results_group)

        detail_group = QGroupBox("Selected Entry Details")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setFont(QFont("Consolas", 10))
        self.detail_view.setPlaceholderText("Select a result to see its full data here.")
        detail_layout.addWidget(self.detail_view)
        main_layout.addWidget(detail_group)
        main_layout.setStretch(1, 1); main_layout.setStretch(2, 1)

    def _connect_signals(self):
        self.search_button.clicked.connect(self._on_search_clicked)
        self.search_input.returnPressed.connect(self._on_search_clicked)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        self.memory_type_combo.currentTextChanged.connect(self._on_memory_type_changed)
        self.importance_slider.valueChanged.connect(self._on_importance_changed)
        self._on_memory_type_changed("Hierarchical")

    def _on_memory_type_changed(self, memory_type: str):
        is_hierarchical = memory_type == "Hierarchical"
        self.hierarchical_filter_frame.setVisible(is_hierarchical)

    def _on_importance_changed(self, value: int):
        self.importance_label.setText(f"Min Importance: {value / 100.0:.2f}")

    def _on_search_clicked(self):
        if not self.api_base_url:
            self.logger.error("Search attempted but API URL is not set.")
            return

        if self._current_search_task and not self._current_search_task.done():
            self.logger.warning("Cancelling previous search task.")
            self._current_search_task.cancel()

        self.search_button.setEnabled(False)
        self.search_button.setText("Searching...")
        self.results_table.setRowCount(0)
        self.results_table.setSortingEnabled(False)
        self.detail_view.clear()
        
        request_payload = {
            "query_text": self.search_input.text().strip(),
            "memory_type": self.memory_type_combo.currentText().lower(),
            "session_id": self.session_id_input.text().strip() or None,
            "min_importance": self.importance_slider.value() / 100.0,
            "limit": 100
        }
        
        self._current_search_task = asyncio.create_task(self._run_search(request_payload))

    async def _run_search(self, payload: Dict[str, Any]):
        self.logger.info(f"Initiating API search in '{payload['memory_type']}' memory.")
        results = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.api_base_url}/api/v1/memory/search", json=payload)
                response.raise_for_status()
                results = response.json().get("results", [])
        except httpx.RequestError as e:
            self.logger.error(f"API request failed during memory search: {e}", exc_info=True)
            self.detail_view.setText(f"API Connection Error:\n\n{e}")
        except asyncio.CancelledError:
            self.logger.info("Search task was cancelled.")
        except Exception as e:
            self.logger.error(f"An error occurred during memory search: {e}", exc_info=True)
            self.detail_view.setText(f"An unexpected error occurred:\n\n{e}")
        finally:
            self.search_button.setEnabled(True)
            self.search_button.setText("Search")
            self._populate_results(results, payload['memory_type'])
            self.results_table.setSortingEnabled(True)
            self._current_search_task = None

    def _populate_results(self, results: List[Dict[str, Any]], memory_type: str):
        self.results_table.setRowCount(len(results))
        if not results: self.logger.info("Search returned no results."); return

        for row_index, result_data in enumerate(results):
            score_val = result_data.get('similarity', result_data.get('confidence'))
            score_item = QTableWidgetItem(f"{score_val:.3f}" if score_val is not None else "N/A")
            type_str = result_data.get('tier', memory_type).upper()
            type_item = QTableWidgetItem(type_str)
            content_val = result_data.get('content', result_data.get('claim', ''))
            snippet = (content_val[:100] + '...') if len(content_val) > 100 else content_val
            content_item = QTableWidgetItem(snippet.replace("\n", " "))
            timestamp_val = result_data.get('timestamp', result_data.get('created_at'))
            ts_str = timestamp_val if isinstance(timestamp_val, str) else "N/A"
            if isinstance(timestamp_val, str):
                try:
                    ts_str = datetime.fromisoformat(timestamp_val.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    ts_str = timestamp_val # Keep original if parsing fails
            timestamp_item = QTableWidgetItem(ts_str)
            importance_val = result_data.get('importance', result_data.get('importance_score'))
            importance_item = QTableWidgetItem(f"{importance_val:.3f}" if importance_val is not None else "N/A")
            score_item.setData(Qt.ItemDataRole.UserRole, result_data)
            self.results_table.setItem(row_index, 0, score_item)
            self.results_table.setItem(row_index, 1, type_item)
            self.results_table.setItem(row_index, 2, content_item)
            self.results_table.setItem(row_index, 3, timestamp_item)
            self.results_table.setItem(row_index, 4, importance_item)
        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

    def _on_result_selected(self):
        selected_items = self.results_table.selectedItems()
        if not selected_items: return
        first_item = self.results_table.item(selected_items[0].row(), 0)
        if not first_item:
            self.detail_view.setText("Error: Could not retrieve data for selected row.")
            return
        full_data = first_item.data(Qt.ItemDataRole.UserRole)
        if full_data:
            try:
                formatted_json = json.dumps(full_data, indent=4, default=str)
                self.detail_view.setText(formatted_json)
            except Exception as e:
                self.logger.error(f"Failed to serialize result data for detail view: {e}", exc_info=True)
                self.detail_view.setText(f"Error displaying details: {e}\n\nRaw data:\n{str(full_data)}")
        else:
             self.detail_view.setText("No detailed data available for this entry.")