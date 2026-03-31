"""
Hardware abstraction for MacBook Pro M4 and Mac Mini 2018.
Provides unified interface for device selection and profiling.
"""
from __future__ import annotations
import os
import subprocess
import platform
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class DeviceType(Enum):
    APPLE_SILICON_M4 = "apple_silicon_m4"
    INTEL_X86_64 = "intel_x86_64"
    CPU_ONLY = "cpu"

@dataclass
class DeviceSpec:
    name: str
    device_type: DeviceType
    total_memory_gb: float
    cpu_cores: int
    has_gpu: bool
    gpu_name: Optional[str] = None
    has_neural_engine: bool = False
    memory_bandwidth_gb_s: Optional[float] = None

class M4Device:
    """MacBook Pro M4 Apple Silicon"""
    @staticmethod
    def spec() -> DeviceSpec:
        return DeviceSpec(
            name="MacBook Pro M4 (Apple Silicon)",
            device_type=DeviceType.APPLE_SILICON_M4,
            total_memory_gb=48.0,
            cpu_cores=14,
            has_gpu=True,
            gpu_name="M4 Pro GPU (10-core)",
            has_neural_engine=True,
            memory_bandwidth_gb_s=120.0,
        )

    @staticmethod
    def is_available() -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @staticmethod
    def recommended_batch_size(model_name: str, model_size_b: int) -> int:
        """Estimate recommended batch size based on model size and available memory."""
        # Leave 10GB for OS + framework overhead
        available_gb = 48.0 - 10.0
        # Rough estimate: 2 bytes per parameter for activations (fp16)
        activations_gb = model_size_b * 2
        total = model_size_b + activations_gb
        if total > available_gb:
            scale = available_gb / total
            return max(1, int(scale))
        return 32  # default

    @staticmethod
    def mlx_preferred() -> bool:
        """MLX is preferred for Apple Silicon."""
        return True

class MacMiniDevice:
    """Mac Mini 2018 Intel"""
    @staticmethod
    def spec() -> DeviceSpec:
        return DeviceSpec(
            name="Mac Mini 2018 (Intel)",
            device_type=DeviceType.INTEL_X86_64,
            total_memory_gb=64.0,
            cpu_cores=6,
            has_gpu=False,
            gpu_name=None,
            has_neural_engine=False,
            memory_bandwidth_gb_s=None,
        )

    @staticmethod
    def is_available() -> bool:
        # Check if we can SSH to Mac Mini (local check for now)
        return True  # Will be overridden by actual connectivity check

    @staticmethod
    def fenics_preferred() -> bool:
        """FEniCS runs natively on Intel."""
        return True

class DeviceManager:
    """
    Manages device selection based on availability and task requirements.
    """
    def __init__(self):
        self.local_device: DeviceSpec = M4Device.spec()
        self.remote_device: Optional[DeviceSpec] = None
        self._detect_remote()

    def _detect_remote(self):
        """Detect if Mac Mini is reachable."""
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", "192.168.1.100"],
                capture_output=True, timeout=2
            )
            # NOTE: Replace with actual Mac Mini IP
            self.remote_device = MacMiniDevice.spec()
        except Exception:
            self.remote_device = None

    def select_for_task(self, task: str) -> DeviceSpec:
        """
        Select appropriate device for a given task.

        Tasks:
          - "pinn_training": M4 (MLX GPU acceleration)
          - "fenics_baseline": Mac Mini (Intel optimized)
          - "large_model_inference": M4 (unified memory)
          - "cfd_openfoam": Mac Mini (more RAM for large cases)
          - "ising_mc": M4 (MLX for Metropolis, or CPU)
        """
        task_map = {
            "pinn_training": self.local_device,
            "deeponet_training": self.local_device,
            "fno_training": self.local_device,
            "fenics_baseline": self.remote_device or self.local_device,
            "large_model_inference": self.local_device,
            "cfd_openfoam": self.remote_device or self.local_device,
            "ising_mc": self.local_device,
            "uq_ensemble": self.local_device,
            "quantized_inference": self.local_device,
        }
        return task_map.get(task, self.local_device)

    def summary(self) -> str:
        lines = [f"Local device: {self.local_device.name}",
                 f"  Memory: {self.local_device.total_memory_gb}GB",
                 f"  CPU cores: {self.local_device.cpu_cores}",
                 f"  Neural Engine: {self.local_device.has_neural_engine}"]
        if self.remote_device:
            lines.append(f"Remote device: {self.remote_device.name} (connected)")
        else:
            lines.append("Remote device: Mac Mini 2018 (not detected)")
        return "\n".join(lines)


if __name__ == "__main__":
    dm = DeviceManager()
    print(dm.summary())
    print()
    for task in ["pinn_training", "fenics_baseline", "ising_mc", "uq_ensemble"]:
        d = dm.select_for_task(task)
        print(f"Task '{task}' → {d.name}")
