# backend/gui/panels/system_panel.py

from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel,
                             QProgressBar, QHBoxLayout, QFrame)
from PyQt6.QtCore import QTimer

from backend.hardware.resource_manager import HardwareResourceManager
from backend.utils.logger import Logger


class SystemPanel(QWidget):
    """
    A GUI panel that displays real-time hardware resource utilization,
    including CPU, RAM, and GPU metrics.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initializes the SystemPanel."""
        super().__init__(parent)
        self.logger = Logger(__name__)
        self.resource_manager: Optional[HardwareResourceManager] = None
        
        # Dictionaries to hold references to dynamically created widgets
        self.gpu_widgets: Dict[int, Dict[str, Any]] = {}
        
        self._setup_ui()
        
        # Set up a timer to periodically request metric updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_metrics)
        
        self.logger.info("SystemPanel initialized.")

    def set_resource_manager(self, manager: HardwareResourceManager):
        """
        Links the panel to the HardwareResourceManager and starts monitoring.
        """
        self.resource_manager = manager
        self.logger.info("SystemPanel linked to HardwareResourceManager.")
        self._populate_static_specs()
        # Start the timer to poll for updates every 2 seconds
        self.update_timer.start(2000)

    def _setup_ui(self):
        """Sets up the static UI elements. Dynamic elements are created later."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- Static Hardware Specifications ---
        specs_group = QGroupBox("Hardware Specifications")
        self.specs_layout = QVBoxLayout(specs_group)
        main_layout.addWidget(specs_group)

        # --- Real-time Usage ---
        usage_group = QGroupBox("Real-Time Resource Usage")
        usage_layout = QVBoxLayout(usage_group)

        # CPU Usage
        self.cpu_usage_label = QLabel("CPU Usage: N/A")
        self.cpu_usage_bar = QProgressBar()
        usage_layout.addWidget(self.cpu_usage_label)
        usage_layout.addWidget(self.cpu_usage_bar)

        # RAM Usage
        self.ram_usage_label = QLabel("RAM Usage: N/A")
        self.ram_usage_bar = QProgressBar()
        usage_layout.addWidget(self.ram_usage_label)
        usage_layout.addWidget(self.ram_usage_bar)
        
        # This layout will be populated with GPU widgets dynamically
        self.gpu_usage_layout = QVBoxLayout()
        self.gpu_usage_layout.setSpacing(10)
        usage_layout.addLayout(self.gpu_usage_layout)

        main_layout.addWidget(usage_group)
        main_layout.addStretch()

    def _populate_static_specs(self):
        """Populates the UI with static hardware info once the manager is available."""
        if not self.resource_manager:
            return

        specs = self.resource_manager.get_hardware_specs()
        
        # Clear any previous specs
        while self.specs_layout.count():
            child = self.specs_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Populate CPU and RAM specs
        cpu_spec_label = QLabel(f"CPU: {specs['cpu_count_physical']} Cores / {specs['cpu_count_logical']} Threads")
        ram_spec_label = QLabel(f"Total RAM: {specs['total_ram_gb']:.2f} GB")
        self.specs_layout.addWidget(cpu_spec_label)
        self.specs_layout.addWidget(ram_spec_label)
        
        # Dynamically create widgets for each GPU
        gpus = specs.get('gpus', [])
        if not gpus:
            self.specs_layout.addWidget(QLabel("GPU: No CUDA-enabled GPU detected."))
            return

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.gpu_usage_layout.addWidget(separator)
            
        for gpu_spec in gpus:
            gpu_id = gpu_spec['id']
            gpu_name = gpu_spec['name']
            gpu_total_mem = gpu_spec['total_memory_gb']
            
            self.specs_layout.addWidget(QLabel(f"GPU {gpu_id}: {gpu_name} ({gpu_total_mem:.2f} GB)"))
            
            # Create the dynamic usage widgets for this GPU
            gpu_load_label = QLabel(f"GPU {gpu_id} Load: N/A")
            gpu_load_bar = QProgressBar()
            
            gpu_mem_label = QLabel(f"GPU {gpu_id} Memory: N/A")
            gpu_mem_bar = QProgressBar()

            self.gpu_usage_layout.addWidget(gpu_load_label)
            self.gpu_usage_layout.addWidget(gpu_load_bar)
            self.gpu_usage_layout.addWidget(gpu_mem_label)
            self.gpu_usage_layout.addWidget(gpu_mem_bar)

            # Store references to these widgets for easy updating
            self.gpu_widgets[gpu_id] = {
                'load_label': gpu_load_label,
                'load_bar': gpu_load_bar,
                'mem_label': gpu_mem_label,
                'mem_bar': gpu_mem_bar,
                'total_mem_gb': gpu_total_mem
            }

    def _update_metrics(self):
        """This slot is called by the QTimer to update the UI with fresh metrics."""
        if not self.resource_manager:
            return
            
        metrics = self.resource_manager.get_current_metrics()
        
        if not metrics:
            return  # No new data available from the queue yet

        # Update CPU display
        cpu_percent = metrics.get('cpu_percent', 0.0)
        self.cpu_usage_bar.setValue(int(cpu_percent))
        self.cpu_usage_label.setText(f"CPU Usage: {cpu_percent:.1f}%")

        # Update RAM display
        ram_info = metrics.get('ram', {})
        ram_percent = ram_info.get('percent', 0.0)
        ram_used_gb = ram_info.get('used_gb', 0.0)
        ram_total_gb = ram_info.get('total_gb', 0.0)
        self.ram_usage_bar.setValue(int(ram_percent))
        self.ram_usage_label.setText(f"RAM Usage: {ram_percent:.1f}% ({ram_used_gb:.2f} / {ram_total_gb:.2f} GB)")

        # Update GPU displays
        for gpu_metric in metrics.get('gpus', []):
            gpu_id = gpu_metric.get('id')
            if gpu_id in self.gpu_widgets:
                widgets = self.gpu_widgets[gpu_id]
                
                # Update load bar
                load_percent = gpu_metric.get('load_percent', 0.0)
                widgets['load_bar'].setValue(int(load_percent))
                widgets['load_label'].setText(f"GPU {gpu_id} Load: {load_percent:.1f}%")
                
                # Update memory bar and label
                mem_percent = gpu_metric.get('memory_percent', 0.0)
                mem_used_gb = gpu_metric.get('memory_used_gb', 0.0)
                total_mem_gb = widgets['total_mem_gb']
                widgets['mem_bar'].setValue(int(mem_percent))
                widgets['mem_label'].setText(f"GPU {gpu_id} Memory: {mem_percent:.1f}% ({mem_used_gb:.2f} / {total_mem_gb:.2f} GB)")