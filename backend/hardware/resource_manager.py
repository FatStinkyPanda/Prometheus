# backend/hardware/resource_manager.py

import sys
import threading
import time
import queue
from typing import Dict, Any, List, Optional

import psutil
import torch

from backend.utils.logger import Logger

# Optional dependency for GPU utilization.
try:
    import GPUtil
except ImportError:
    GPUtil = None


class HardwareResourceManager:
    """
    A Singleton class for monitoring and managing hardware resources (CPU, RAM, GPU).

    This manager runs a background thread to periodically poll for resource usage,
    making the latest metrics available to the application in a thread-safe manner
    without blocking the main event loop.
    """
    _instance: Optional['HardwareResourceManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(HardwareResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes the HardwareResourceManager."""
        if hasattr(self, '_initialized'):
            return
            
        self.logger = Logger(__name__)
        
        # --- Static Hardware Specs ---
        self.cpu_count_physical = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.gpu_specs = self._detect_gpus()

        # --- Monitoring Thread ---
        self.monitoring_interval_sec = 2.0
        self._monitoring_thread: Optional[threading.Thread] = None
        self._is_monitoring = threading.Event()
        self.metrics_queue = queue.Queue(maxsize=1) # Only store the latest metric

        self._initialized = True
        self.logger.info("HardwareResourceManager initialized.")
        self.log_hardware_summary()

    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Detects available GPUs and their static properties."""
        specs = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                specs.append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024 ** 3)
                })
        return specs

    def log_hardware_summary(self):
        """Logs a summary of the detected hardware."""
        self.logger.info("--- Hardware Summary ---")
        self.logger.info(f"CPU: {self.cpu_count_physical} Physical Cores, {self.cpu_count_logical} Logical Processors")
        self.logger.info(f"RAM: {self.total_ram_gb:.2f} GB Total")
        if self.gpu_specs:
            for gpu in self.gpu_specs:
                self.logger.info(f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.2f} GB)")
        else:
            self.logger.info("GPU: No CUDA-enabled GPU detected.")
        if not GPUtil:
            self.logger.warning("GPUtil library not found. GPU load metrics will be unavailable.")
        self.logger.info("------------------------")

    def start_monitoring(self):
        """Starts the background resource monitoring thread."""
        if self._is_monitoring.is_set():
            self.logger.warning("Monitoring is already active.")
            return

        self.logger.info("Starting hardware monitoring thread...")
        self._is_monitoring.set()
        self._monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stops the background resource monitoring thread."""
        if not self._is_monitoring.is_set():
            self.logger.info("Monitoring is not active.")
            return

        self.logger.info("Stopping hardware monitoring thread...")
        self._is_monitoring.clear()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=self.monitoring_interval_sec * 2)
        self.logger.info("Hardware monitoring stopped.")

    def _monitor_loop(self):
        """The main loop for the monitoring thread."""
        while self._is_monitoring.is_set():
            try:
                metrics = self._poll_system_metrics()
                # Update the queue, discarding the old item if one exists
                if not self.metrics_queue.full():
                    self.metrics_queue.put_nowait(metrics)
                else:
                    try:
                        self.metrics_queue.get_nowait() # Discard old
                        self.metrics_queue.put_nowait(metrics) # Put new
                    except queue.Empty:
                        pass # Race condition, another thread got it, which is fine.

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            
            time.sleep(self.monitoring_interval_sec)

    def _poll_system_metrics(self) -> Dict[str, Any]:
        """Polls and returns a snapshot of the current system metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent()
        
        # RAM
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used_gb = ram.used / (1024 ** 3)
        
        # GPUs
        gpu_metrics = []
        if self.gpu_specs:
            gputil_gpus = GPUtil.getGPUs() if GPUtil else []
            for gpu_spec in self.gpu_specs:
                gpu_id = gpu_spec['id']
                
                # Utilization from GPUtil if available
                load_percent = 0.0
                if len(gputil_gpus) > gpu_id:
                    load_percent = gputil_gpus[gpu_id].load * 100

                # Memory from PyTorch
                try:
                    mem_info = torch.cuda.mem_get_info(gpu_id)
                    mem_total_gb = mem_info[1] / (1024 ** 3)
                    mem_free_gb = mem_info[0] / (1024 ** 3)
                    mem_used_gb = mem_total_gb - mem_free_gb
                    mem_percent = (mem_used_gb / mem_total_gb) * 100 if mem_total_gb > 0 else 0
                except Exception as e:
                     self.logger.warning(f"Could not get memory info for GPU {gpu_id}. Error: {e}")
                     mem_percent, mem_used_gb = 0.0, 0.0

                gpu_metrics.append({
                    "id": gpu_id,
                    "load_percent": load_percent,
                    "memory_percent": mem_percent,
                    "memory_used_gb": mem_used_gb
                })

        return {
            "cpu_percent": cpu_percent,
            "ram": {"percent": ram_percent, "used_gb": ram_used_gb, "total_gb": self.total_ram_gb},
            "gpus": gpu_metrics
        }

    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Returns the latest available hardware metrics.
        
        Returns:
            A dictionary of the latest metrics, or None if monitoring is not active
            and no metrics are available.
        """
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            # If the queue is empty, it means the polling thread hasn't run yet
            # or we missed an update. We can return None or do an on-demand poll.
            # On-demand poll is more responsive for the first call.
            if self._is_monitoring.is_set():
                return self._poll_system_metrics()
            return None

    def get_hardware_specs(self) -> Dict[str, Any]:
        """Returns the static hardware specifications detected at startup."""
        return {
            "cpu_count_physical": self.cpu_count_physical,
            "cpu_count_logical": self.cpu_count_logical,
            "total_ram_gb": self.total_ram_gb,
            "gpus": self.gpu_specs
        }

    def cleanup(self):
        """Cleans up resources, primarily stopping the monitoring thread."""
        self.stop_monitoring()


if __name__ == '__main__':
    # A simple self-test to demonstrate functionality
    print("--- Running HardwareResourceManager Self-Test ---")
    
    manager = HardwareResourceManager()
    manager.start_monitoring()
    
    print("Monitoring started. Will print metrics every 2 seconds for 10 seconds.")
    
    try:
        for i in range(5):
            time.sleep(2)
            metrics = manager.get_current_metrics()
            if metrics:
                print(f"\n--- METRICS @ T+{ (i+1)*2 }s ---")
                print(f"CPU Usage: {metrics['cpu_percent']:.1f}%")
                print(f"RAM Usage: {metrics['ram']['percent']:.1f}% ({metrics['ram']['used_gb']:.2f}/{metrics['ram']['total_gb']:.2f} GB)")
                if metrics['gpus']:
                    for gpu in metrics['gpus']:
                        print(f"GPU {gpu['id']} Load: {gpu['load_percent']:.1f}% | Memory: {gpu['memory_percent']:.1f}% ({gpu['memory_used_gb']:.2f} GB used)")
                else:
                    print("No GPUs to report.")
            else:
                print("Waiting for first metric poll...")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        manager.cleanup()
        print("\n--- Self-Test Complete ---")