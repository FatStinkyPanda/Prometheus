# backend/gui/panels/minds_panel.py

from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel,
                             QProgressBar, QTextEdit, QHBoxLayout, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from backend.utils.logger import Logger

class MindsPanel(QWidget):
    """
    A GUI panel for visualizing the real-time state of the AI's cognitive processes,
    including the three specialized minds and the new truth evaluation cycle.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initializes the MindsPanel."""
        super().__init__(parent)
        self.logger = Logger(__name__)
        
        self.emotion_widgets: Dict[str, Dict[str, Any]] = {}
        
        self._setup_ui()
        self.clear_displays()
        self.logger.info("MindsPanel initialized.")

    def _setup_ui(self):
        """Sets up all the UI elements within the panel."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- Logical Mind Group ---
        logical_group = QGroupBox("Logical Mind")
        logical_layout = QVBoxLayout(logical_group)
        self.logical_confidence_bar = QProgressBar()
        self.logical_confidence_bar.setFormat("Confidence: %p%")
        logical_layout.addWidget(self.logical_confidence_bar)
        main_layout.addWidget(logical_group)

        # --- Creative Mind Group ---
        creative_group = QGroupBox("Creative Mind")
        creative_layout = QVBoxLayout(creative_group)
        self.creative_confidence_bar = QProgressBar()
        self.creative_confidence_bar.setFormat("Confidence: %p%")
        self.creative_text_display = QTextEdit()
        self.creative_text_display.setReadOnly(True)
        self.creative_text_display.setPlaceholderText("The final text output will appear here...")
        self.creative_text_display.setMaximumHeight(150)
        self.creative_text_display.setFont(QFont("Segoe UI", 9))
        creative_layout.addWidget(self.creative_confidence_bar)
        creative_layout.addWidget(QLabel("Final Refined Text:"))
        creative_layout.addWidget(self.creative_text_display)
        main_layout.addWidget(creative_group)

        # --- Emotional Mind Group ---
        emotional_group = QGroupBox("Emotional Mind")
        emotional_layout = QVBoxLayout(emotional_group)
        self.emotional_confidence_bar = QProgressBar()
        self.emotional_confidence_bar.setFormat("Dominant Emotion Confidence: %p%")
        self.dominant_emotion_label = QLabel("Dominant Emotion: N/A")
        font = self.dominant_emotion_label.font(); font.setBold(True)
        self.dominant_emotion_label.setFont(font)
        emotional_layout.addWidget(self.emotional_confidence_bar)
        emotional_layout.addWidget(self.dominant_emotion_label)
        separator1 = QFrame(); separator1.setFrameShape(QFrame.Shape.HLine); separator1.setFrameShadow(QFrame.Shadow.Sunken)
        emotional_layout.addWidget(separator1)
        self.emotion_distribution_layout = QVBoxLayout(); self.emotion_distribution_layout.setSpacing(5)
        emotional_layout.addLayout(self.emotion_distribution_layout)
        main_layout.addWidget(emotional_group)

        # --- NEW: Truth Evaluation Group ---
        truth_group = QGroupBox("Truth Evaluation & Learning")
        truth_group.setToolTip("Shows how the AI's proposed response compares to its stored beliefs.")
        truth_layout = QVBoxLayout(truth_group)
        self.truth_status_label = QLabel("STATUS: N/A")
        font = self.truth_status_label.font(); font.setPointSize(10); font.setBold(True)
        self.truth_status_label.setFont(font)
        self.truth_claim_label = QLabel("Evaluated Claim: N/A")
        self.truth_claim_label.setWordWrap(True)
        self.truth_related_label = QLabel("Related Belief: N/A")
        self.truth_related_label.setWordWrap(True)
        truth_layout.addWidget(self.truth_status_label)
        truth_layout.addWidget(self.truth_claim_label)
        truth_layout.addWidget(self.truth_related_label)
        main_layout.addWidget(truth_group)

        main_layout.addStretch()

    def clear_displays(self):
        """Resets all display widgets to their initial, empty state."""
        self.logger.debug("Clearing mind displays to initial state.")
        self.logical_confidence_bar.setValue(0)
        self.creative_confidence_bar.setValue(0)
        self.creative_text_display.setPlaceholderText("Waiting for new interaction...")
        self.creative_text_display.clear()
        self.emotional_confidence_bar.setValue(0)
        self.dominant_emotion_label.setText("Dominant Emotion: N/A")
        
        self.truth_status_label.setText("STATUS: N/A")
        self.truth_status_label.setStyleSheet("color: black;")
        self.truth_claim_label.setText("Evaluated Claim: N/A")
        self.truth_related_label.setText("Related Belief: N/A")
        self.truth_related_label.setVisible(False)

        for widgets in self.emotion_widgets.values():
            while widgets['hbox'].count():
                item = widgets['hbox'].takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            widgets['hbox'].deleteLater()
        self.emotion_widgets.clear()

    def update_displays(self, payload: Optional[Dict[str, Any]]):
        """Public slot to update all panel displays from the full consciousness cycle payload."""
        self.clear_displays()
        
        if not payload or not isinstance(payload, dict):
            self.logger.warning("update_displays called with invalid or empty payload.")
            return
            
        states = payload.get('final_states', {})
        if not states or not isinstance(states, dict):
            self.logger.debug("No final_states in payload to display.")
            return

        self.logger.debug("Updating mind state displays with new data.")
        self._update_minds_display(states)
        self._update_truth_evaluation_display(payload.get('truth_evaluation', {}))

    def _update_minds_display(self, states: Dict[str, Any]):
        """Updates the sections for the three core minds."""
        logical_state = states.get('logical', {})
        if isinstance(logical_state, dict):
            self.logical_confidence_bar.setValue(int(logical_state.get('confidence', 0.0) * 100))

        creative_state = states.get('creative', {})
        if isinstance(creative_state, dict):
            self.creative_confidence_bar.setValue(int(creative_state.get('confidence', 0.0) * 100))
            payload = creative_state.get('payload', {})
            self.creative_text_display.setText(payload.get('generated_text', "[No text generated]"))

        emotional_state = states.get('emotional', {})
        if isinstance(emotional_state, dict):
            self.emotional_confidence_bar.setValue(int(emotional_state.get('confidence', 0.0) * 100))
            payload = emotional_state.get('payload', {})
            if payload and isinstance(payload, dict):
                try:
                    dominant_emotion = max(payload, key=payload.get)
                    self.dominant_emotion_label.setText(f"Dominant Emotion: {dominant_emotion.capitalize()}")
                except (ValueError, TypeError):
                    self.dominant_emotion_label.setText("Dominant Emotion: Error")
                
                sorted_emotions = sorted(payload.items(), key=lambda item: item[1], reverse=True)
                for emotion, score in sorted_emotions:
                    if emotion not in self.emotion_widgets:
                        hbox = QHBoxLayout()
                        label = QLabel(f"{emotion.capitalize()}:"); label.setMinimumWidth(80)
                        bar = QProgressBar(); bar.setFormat("%p%"); bar.setRange(0, 100); bar.setTextVisible(True)
                        hbox.addWidget(label); hbox.addWidget(bar)
                        self.emotion_distribution_layout.addLayout(hbox)
                        self.emotion_widgets[emotion] = {'hbox': hbox, 'label': label, 'bar': bar}
                    self.emotion_widgets[emotion]['bar'].setValue(int(score * 100))

    def _update_truth_evaluation_display(self, truth_data: Dict[str, Any]):
        """Updates the new truth evaluation section."""
        if not truth_data or not isinstance(truth_data, dict):
            self.truth_status_label.setText("STATUS: NOT EVALUATED")
            self.truth_status_label.setStyleSheet("color: #808080;") # Grey
            self.truth_claim_label.setText("No claim was evaluated.")
            return

        status = truth_data.get('status', 'unknown').upper()
        claim = truth_data.get('evaluated_claim', 'N/A')
        related = truth_data.get('related_truth')

        color_map = {
            'NOVEL': ("#1E88E5", "Learning: Statement is new information."),  # Blue
            'CONSISTENT': ("#43A047", "Consistent with existing beliefs."),  # Green
            'CONTRADICTORY': ("#F4511E", "Self-Correction: Contradicts belief."), # Orange-Red
            'UNCERTAIN': ("#FDD835", "Uncertain relative to beliefs."), # Yellow
            'UNKNOWN': ("#616161", "Unknown evaluation status.") # Grey
        }
        color, status_text = color_map.get(status, color_map['UNKNOWN'])
        
        self.truth_status_label.setText(f"STATUS: {status} ({status_text})")
        self.truth_status_label.setStyleSheet(f"color: {color};")
        self.truth_claim_label.setText(f"Evaluated Claim: \"{claim}\"")

        if related and isinstance(related, dict):
            related_claim = related.get('claim', 'N/A')
            related_value = related.get('value', 'N/A')
            self.truth_related_label.setText(f"Related Belief: \"{related_claim}\" is {related_value}.")
            self.truth_related_label.setVisible(True)
        else:
            self.truth_related_label.setVisible(False)