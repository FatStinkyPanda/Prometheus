# backend/gui/widgets/thought_stream.py

from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QGroupBox)
from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QColor, QFont

from backend.utils.logger import Logger

class ThoughtStream(QWidget):
    """
    A widget that displays a real-time stream of thoughts, dreams, and other
    autonomous events from the consciousness engine.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initializes the ThoughtStream widget."""
        super().__init__(parent)
        self.logger = Logger(__name__)
        self.max_items = 250  # Limit the number of items to prevent high memory usage
        
        self._setup_ui()
        self.logger.info("ThoughtStream widget initialized.")

    def _setup_ui(self):
        """Sets up the UI elements for the thought stream."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        group_box = QGroupBox("Autonomous Event Stream")
        group_layout = QVBoxLayout(group_box)
        
        self.stream_list_widget = QListWidget()
        self.stream_list_widget.setWordWrap(True)
        self.stream_list_widget.setFont(QFont("Consolas", 10))
        
        group_layout.addWidget(self.stream_list_widget)
        main_layout.addWidget(group_box)

    def add_event(self, event_data: Dict[str, Any]):
        """
        Public slot to add a new event to the stream display.
        This method is designed to be connected to a Qt signal.

        Args:
            event_data (Dict[str, Any]): A dictionary containing the event details.
                                         Expected keys: 'type' and 'content'.
        """
        if not isinstance(event_data, dict):
            self.logger.warning(f"ThoughtStream received invalid event data of type {type(event_data)}")
            return
            
        event_type = event_data.get('type', 'unknown').upper()
        content = event_data.get('content', 'No content provided.')
        
        # --- Prepare the list item ---
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        
        # Clean up backend-specific prefixes for a cleaner UI display
        prefixes_to_remove = ['[THINKING]:', '[DREAM STORED]:', '[MAINTENANCE]:', '[MAINTENANCE_ERROR]:']
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content.replace(prefix, '').strip()

        display_text = f"[{timestamp}] [{event_type}] {content}"
        item = QListWidgetItem(display_text)

        # --- Style the item based on its type ---
        color_map = {
            'THOUGHT': QColor("#005cb2"),            # Deep blue
            'DREAM': QColor("#6a1b9a"),              # Deep purple
            'MAINTENANCE': QColor("#00838F"),        # Teal/Cyan
            'MAINTENANCE_ERROR': QColor("#D84315"),  # Deep orange/red
            'UNKNOWN': QColor("#616161")             # Grey for unknown types
        }
        item.setForeground(color_map.get(event_type, color_map['UNKNOWN']))
            
        self.stream_list_widget.addItem(item)
        
        # --- Memory Management and Scrolling ---
        if self.stream_list_widget.count() > self.max_items:
            self.stream_list_widget.takeItem(0)
            
        self.stream_list_widget.scrollToBottom()

# Self-test block to demonstrate the widget's functionality
if __name__ == '__main__':
    import sys
    import random
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    
    main_win = QMainWindow()
    main_win.setWindowTitle("ThoughtStream Widget Test")
    main_win.setGeometry(300, 300, 800, 600)
    
    thought_stream_widget = ThoughtStream()
    main_win.setCentralWidget(thought_stream_widget)
    
    def simulate_backend_event():
        event_type = random.choice(['thought', 'dream', 'maintenance', 'maintenance_error'])
        if event_type == 'thought':
            content = f"Pondering memory: 'User: What is the capital of France? ...' (id: {random.randint(1000,9999)})"
        elif event_type == 'dream':
            content = f"'{random.choice(['A river of clocks', 'A forest of whispers', 'A city made of glass'])}...' (id: {random.randint(1000,9999)})"
        elif event_type == 'maintenance':
            content = f"Promoting {random.randint(5,20)} nodes from ACTIVE to RECENT tier."
        else: # maintenance_error
            content = "Failed to decompress chunk data. File may be corrupted."
            
        event_data = {'type': event_type, 'content': content}
        thought_stream_widget.add_event(event_data)

    timer = QTimer()
    timer.timeout.connect(simulate_backend_event)
    timer.start(2000)
    
    main_win.show()
    sys.exit(app.exec())