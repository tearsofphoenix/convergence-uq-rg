"""
Safe launcher for the Paper 3 main benchmark.

Priority:
  1. Use the native MLX implementation when MLX can initialize successfully.
  2. Fall back to the PyTorch mirror implementation when MLX crashes during
     Metal device discovery in the current session.

This keeps the command entrypoint stable:

    python topic3_rg/cross_scale_mlx.py

while preventing the hard process abort that occurs when `import mlx.core`
crashes before Python can handle an exception.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_runtime_cache() -> None:
    mpl_dir = ROOT / ".runtime-cache" / "matplotlib"
    xdg_dir = ROOT / ".runtime-cache" / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


def probe_mlx() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-c",
        "import mlx.core as mx; print('mlx_probe_ok'); print(mx.default_device())",
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, result.stdout.strip()
    message = (result.stderr or result.stdout).strip()
    return False, message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Paper 3 main benchmark with MLX-or-PyTorch fallback.")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_cache()
    ok, detail = probe_mlx()
    if ok:
        print("[launcher] MLX probe succeeded. Running native MLX implementation.", flush=True)
        from topic3_rg import cross_scale_mlx_impl as impl

        impl.print_protocol_banner()
        impl.run(
            n_seeds=args.n_seeds,
            n_train=args.n_train,
            n_test=args.n_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        return

    print("[launcher] MLX probe failed. Falling back to PyTorch mirror implementation.", flush=True)
    if detail:
        print("[launcher] MLX failure summary:", flush=True)
        print(detail, flush=True)

    from topic3_rg.cross_scale_experiment import (
        print_protocol_banner as torch_banner,
        run_cross_scale_experiment,
    )

    torch_banner()
    run_cross_scale_experiment(
        n_seeds=args.n_seeds,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
