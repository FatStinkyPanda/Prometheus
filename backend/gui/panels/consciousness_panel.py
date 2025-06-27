# backend/gui/panels/consciousness_panel.py

from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel,
                             QPushButton, QListWidget, QListWidgetItem, QHBoxLayout,
                             QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QDateTime
from PyQt6.QtGui import QFont, QColor

from backend.core.consciousness.unified_consciousness import UnifiedConsciousness, ConsciousnessState
from backend.utils.logger import Logger


class ConsciousnessPanel(QWidget):
    """
    A GUI panel that displays the current state of the consciousness and provides
    controls for its autonomous functions. This version is updated to show the
    new 'MAINTAINING' state.
    """
    # Signals emitted when a user clicks a control button
    start_thinking_requested = pyqtSignal()
    start_dreaming_requested = pyqtSignal()
    stop_autonomous_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the ConsciousnessPanel.
        """
        super().__init__(parent)
        self.logger = Logger(__name__)
        self.consciousness: Optional[UnifiedConsciousness] = None
        
        self._setup_ui()
        self._connect_signals()
        self.set_initial_state()
        self.logger.info("ConsciousnessPanel initialized.")

    def set_consciousness_instance(self, consciousness: UnifiedConsciousness):
        """
        Provides the panel with a reference to the main consciousness engine.
        """
        self.consciousness = consciousness
        self.logger.info("ConsciousnessPanel linked to consciousness instance.")
        self.update_state(consciousness.state)
        self._set_controls_enabled(True)

    def _setup_ui(self):
        """Sets up all the UI elements within the panel."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # --- State Display Group ---
        state_group = QGroupBox("Consciousness State")
        state_group_layout = QVBoxLayout(state_group)
        
        self.state_label = QLabel("OFFLINE")
        font = self.state_label.font()
        font.setPointSize(20)
        font.setBold(True)
        self.state_label.setFont(font)
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.state_label.setFrameShadow(QFrame.Shadow.Sunken)
        self.state_label.setMinimumHeight(60)
        
        state_group_layout.addWidget(self.state_label)
        layout.addWidget(state_group)

        # --- Autonomous Controls Group ---
        controls_group = QGroupBox("Autonomous Functions")
        controls_layout = QHBoxLayout(controls_group)
        
        self.think_button = QPushButton("Start Thinking")
        self.think_button.setToolTip("Begin autonomous reflection and analysis.")
        self.think_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        self.dream_button = QPushButton("Start Dreaming")
        self.dream_button.setToolTip("Begin creative synthesis and memory consolidation.")
        self.dream_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip("Stop the current autonomous process.")
        self.stop_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        controls_layout.addWidget(self.think_button)
        controls_layout.addWidget(self.dream_button)
        controls_layout.addWidget(self.stop_button)
        layout.addWidget(controls_group)

        # --- Event Log Group ---
        log_group = QGroupBox("State Event Log")
        log_group_layout = QVBoxLayout(log_group)
        
        self.event_log = QListWidget()
        self.event_log.setWordWrap(True)
        log_group_layout.addWidget(self.event_log)
        layout.addWidget(log_group)

        layout.addStretch() # Pushes content to the top

    def _connect_signals(self):
        """Connects UI signals to appropriate slots or signal emissions."""
        self.think_button.clicked.connect(self.start_thinking_requested.emit)
        self.dream_button.clicked.connect(self.start_dreaming_requested.emit)
        self.stop_button.clicked.connect(self.stop_autonomous_requested.emit)

    def set_initial_state(self):
        """Sets the UI to its default, disconnected state."""
        self.update_state(ConsciousnessState.OFFLINE)
        self.add_log_entry("Panel initialized. Waiting for backend connection...", "info")
        self._set_controls_enabled(False)
        self.stop_button.setEnabled(False)

    def _set_controls_enabled(self, enabled: bool):
        """Enables or disables the main control buttons."""
        self.think_button.setEnabled(enabled)
        self.dream_button.setEnabled(enabled)
        # The stop button is managed separately by the autonomous mode state

    def update_state(self, state: ConsciousnessState):
        """
        Public slot to update the displayed consciousness state and its style.
        """
        self.state_label.setText(state.name.replace("_", " ").title())
        
        color_map = {
            ConsciousnessState.OFFLINE: "#808080",
            ConsciousnessState.INITIALIZING: "#FFA500",
            ConsciousnessState.ACTIVE: "#00C853",
            ConsciousnessState.CONVERSING: "#2979FF",
            ConsciousnessState.THINKING: "#6200EA",
            ConsciousnessState.DREAMING: "#D500F9",
            ConsciousnessState.MAINTAINING: "#00B8D4", # NEW: A cyan color for maintenance
            ConsciousnessState.SHUTTING_DOWN: "#D50000",
        }
        background_color = color_map.get(state, "#FFFFFF")
        # Use white text for dark backgrounds, black for light ones.
        text_color = "#FFFFFF" if state not in [ConsciousnessState.INITIALIZING, ConsciousnessState.ACTIVE] else "#000000"
        
        self.state_label.setStyleSheet(
            f"background-color: {background_color}; color: {text_color}; border-radius: 5px; padding: 5px;"
        )
        self.add_log_entry(f"State changed to: {state.name}", "state_change")

    def set_autonomous_mode(self, is_active: bool, mode: Optional[str] = None):
        """
        Updates the UI to reflect an active autonomous state (Thinking/Dreaming).
        
        Args:
            is_active (bool): True if an autonomous mode is starting, False if stopping.
            mode (Optional[str]): The name of the mode ('Thinking' or 'Dreaming').
        """
        if is_active:
            self.logger.info(f"Setting autonomous mode active for '{mode}'.")
            self.think_button.setEnabled(False)
            self.dream_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            if mode:
                self.add_log_entry(f"Autonomous {mode} started.", "action")
        else:
            self.logger.info("Setting autonomous mode inactive.")
            # Only enable the buttons if the system is in a state that allows it.
            if self.consciousness and self.consciousness.state == ConsciousnessState.ACTIVE:
                 self._set_controls_enabled(True)
            else:
                 self._set_controls_enabled(False)
            
            self.stop_button.setEnabled(False)
            self.add_log_entry("Autonomous process stopped.", "action")
            
            # Revert to base state if consciousness object is available
            if self.consciousness:
                self.update_state(self.consciousness.state)

    def add_log_entry(self, message: str, log_type: str = "info"):
        """
        Adds a formatted entry to the event log.
        
        Args:
            message (str): The log message content.
            log_type (str): The type of log ('info', 'state_change', 'action', 'error').
        """
        now_str = QDateTime.currentDateTime().toString("HH:mm:ss")
        
        item = QListWidgetItem(f"[{now_str}] {message}")
        
        color_map = {
            "info": QColor("#616161"),
            "state_change": QColor("#6200EA"), # Using the THINKING color for state changes
            "action": QColor("#00838F"),
            "maintenance": QColor("#00B8D4"), # Using the MAINTAINING color
            "maintenance_error": QColor("#BF360C"),
            "error": QColor("#D50000"),
        }
        item.setForeground(color_map.get(log_type, QColor("#000000")))
        
        self.event_log.addItem(item)
        self.event_log.scrollToBottom()
        
        # Limit the number of items in the log to prevent memory issues
        if self.event_log.count() > 200:
            self.event_log.takeItem(0)