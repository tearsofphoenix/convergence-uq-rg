# Shared Module

Utilities used across all three research topics.

## PDE Test Suite (`pde_suite.py`)

Canonical PDEs for benchmarking:
- 13 test cases across 5 categories
- Analytical solutions where available
- Difficulty ratings for resource planning

## Hardware Abstraction (`hardware.py`)

Unified device management:
- M4 Pro (48GB): MLX-accelerated training, neural engine tasks
- Mac Mini 2018 (64GB): FEniCS baselines, OpenFOAM CFD

## Data Manager (`data_manager.py`)

Dataset versioning and storage management.
