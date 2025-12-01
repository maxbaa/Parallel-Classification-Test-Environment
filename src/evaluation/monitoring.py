"""
monitoring.py

This module provides monitoring utilities for experiments:
- CPU usage tracking
- RAM usage tracking
- GPU usage tracking (optional, if GPU is available)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
import psutil
from typing import List
import numpy as np
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ResourceSnapshot:
    """Data class to hold resource usage statistics."""
    cpu_avg: float
    cpu_max: float
    ram_avg_gb: float
    ram_max_gb: float
    gpu_max_gb: float

@dataclass
class ResourceMonitor:
    """
    Class for monitoring system resources during experiments.
    Uses background thread to periodically sample CPU, RAM, and GPU usage.
    """

    def __init__(self, interval:float=0.1, enable_gpu:bool=True):
        self.interval = interval
        self.enable_gpu = enable_gpu
        
        self._stop_event = threading.Event()
        self._thread = None
        
        self._cpu_samples: List[float] = []
        self._ram_samples: List[float] = []
        self._gpu_max_gb: float = 0.0

    def _poll(self):
        process = psutil.Process()
        process.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            cpu = process.cpu_percent(interval=None)
            ram = process.memory_info().rss / (1024 ** 3)

            self._cpu_samples.append(cpu)
            self._ram_samples.append(ram)

            if self.enable_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                current_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                self._gpu_max_gb = max(self._gpu_max_gb, current_alloc)

            time.sleep(self.interval)

    def __enter__(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

        if self.enable_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        
        if self._thread is not None:
            self._thread.join()

        if self.enable_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self._gpu_max_gb = max(self._gpu_max_gb, peak)

    def get_snapshot(self) -> ResourceSnapshot:
        cpu_avg = np.mean(self._cpu_samples) if self._cpu_samples else 0.0
        cpu_max = np.max(self._cpu_samples) if self._cpu_samples else 0
        ram_avg = np.mean(self._ram_samples) if self._ram_samples else 0.0
        ram_max = np.max(self._ram_samples) if self._ram_samples else 0.0
        
        return ResourceSnapshot(
            cpu_avg=cpu_avg,
            cpu_max=cpu_max,
            ram_avg_gb=ram_avg,
            ram_max_gb=ram_max,
            gpu_max_gb=self._gpu_max_gb
        )