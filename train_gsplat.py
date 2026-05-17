#!/usr/bin/env python3
"""
GSplat training script — paleidziamas is RunPod workerio.

Priima COLMAP output'a (text formata) ir treniruoja 3DGS modeli.
Naudoja gsplat simple_trainer is https://github.com/nerfstudio-project/gsplat
"""

import argparse
import os
import sys
from pathlib import Path

# Pridedam gsplat/examples i path'a
GS_DIR = Path('/app/gsplat')
sys.path.insert(0, str(GS_DIR / 'examples'))

from simple_trainer import SimpleTrainer, DefaultConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', required=True, help='Images directory')
    parser.add_argument('--model_path', required=True, help='Output directory')
    parser.add_argument('--colmap_path', required=True, help='COLMAP text export directory')
    parser.add_argument('--iterations', type=int, default=15000)
    parser.add_argument('--save_ply', type=int, default=1)
    args = parser.parse_args()

    source_path = Path(args.source_path)
    colmap_path = Path(args.colmap_path)
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_path}")
    print(f"COLMAP: {colmap_path}")
    print(f"Output: {model_path}")
    print(f"Iterations: {args.iterations}")

    # Konfigūracija
    config = DefaultConfig()
    config.data_dir = str(source_path)
    config.result_dir = str(model_path)
    config.disable_viewer = True
    config.eval_steps = -1  # No eval during training
    config.data_factor = 1  # Full resolution
    config.max_steps = args.iterations
    config.save_ply = bool(args.save_ply)

    print(f"Config: {config.__dict__}")
    print("Pradedamas SimpleTrainer treniravimas...")

    trainer = SimpleTrainer(config)
    trainer.train()

    print(f"✓ Treniravimas baigtas. Rezultatai: {model_path}")


if __name__ == '__main__':
    main()
