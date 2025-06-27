# backend/gui/launcher_window.py

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton,
                             QHBoxLayout, QFrame, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class LauncherWindow(QDialog):
    """
    A simple dialog window that appears on application startup, allowing the user
    to choose which mode to launch.
    """
    # Signal that is emitted when a launch option is chosen.
    # The string argument will be 'backend' or 'frontend'.
    launch_option_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prometheus Launcher")
        self.setMinimumSize(400, 250)
        self.setModal(True) # Acts like a proper dialog window
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Creates the UI elements for the launcher window."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Title Label ---
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label = QLabel("Welcome to Prometheus")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # --- Description Label ---
        desc_font = QFont()
        desc_font.setPointSize(10)
        desc_label = QLabel("Please select an operating mode:")
        desc_label.setFont(desc_font)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(desc_label)
        
        main_layout.addSpacing(10)
        
        # --- Buttons ---
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        # Backend Mode Button
        self.backend_button = QPushButton("Launch Backend GUI (Phase 1)")
        self.backend_button.setMinimumHeight(45)
        self.backend_button.setToolTip("Start the full-featured backend with the PyQt6 desktop interface.")
        font = self.backend_button.font()
        font.setPointSize(11)
        self.backend_button.setFont(font)

        # Frontend Mode Button
        self.frontend_button = QPushButton("Launch Full Stack (Phase 2)")
        self.frontend_button.setMinimumHeight(45)
        self.frontend_button.setToolTip("Start the backend API and launch the React frontend. (Not yet implemented)")
        self.frontend_button.setFont(font)
        self.frontend_button.setEnabled(False) # Disabled for now

        button_layout.addWidget(self.backend_button)
        button_layout.addWidget(self.frontend_button)
        
        main_layout.addLayout(button_layout)
        
    def _connect_signals(self):
        """Connects button clicks to the signal emission."""
        self.backend_button.clicked.connect(self._on_backend_selected)
        self.frontend_button.clicked.connect(self._on_frontend_selected)

    def _on_backend_selected(self):
        """Emits the 'backend' signal and closes the dialog."""
        self.launch_option_selected.emit("backend")
        self.accept() # QDialog's way of closing and returning a "success" result

    def _on_frontend_selected(self):
        """Emits the 'frontend' signal and closes the dialog."""
        self.launch_option_selected.emit("frontend")
        self.accept()

# Self-test block to see how the window looks
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    launcher = LauncherWindow()
    
    # Define a simple slot to show which option was chosen
    def show_choice(option):
        print(f"Launch option chosen: {option}")
        
    launcher.launch_option_selected.connect(show_choice)
    
    launcher.exec() # Use exec() for modal dialogs
    sys.exit(0)