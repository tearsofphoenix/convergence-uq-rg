"""
Data manager for versioning and sharing large datasets between M4 and Mac Mini.
Uses local filesystem + optional rsync for network transfer.
"""
from __future__ import annotations
import os
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DatasetSpec:
    name: str
    size_gb: float
    path: str
    md5: Optional[str] = None

class DataManager:
    """
    Manages datasets for all three research topics.
    Datasets stored on Mac Mini's external SSD, accessed from M4 via network share.
    """
    def __init__(self, base_path: str = "/tmp/research-data"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.datasets = {}

    def register(self, spec: DatasetSpec):
        self.datasets[spec.name] = spec

    def path(self, name: str) -> Path:
        return self.base / self.datasets[name].path

    def register_all(self):
        # Topic 1: Neural Operator Benchmark
        self.register(DatasetSpec("NeuralOpBench", 5.0, "topic1/neural_op_bench"))
        self.register(DatasetSpec("PDE-Bench-UQ", 10.0, "topic2/pde_bench_uq"))
        self.register(DatasetSpec("PhyRG-Bench", 200.0, "topic3/phy_rg_bench"))

        # Additional shared datasets
        self.register(DatasetSpec("fenics_reference_solutions", 2.0, "shared/fenics_ref"))
        self.register(DatasetSpec("openfoam_cases", 50.0, "shared/openfoam"))

    def disk_usage(self) -> str:
        total = sum(d.size_gb for d in self.datasets.values())
        return f"Total datasets: {len(self.datasets)}, ~{total}GB"
