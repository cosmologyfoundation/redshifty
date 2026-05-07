"""
Smoke Test: Quick 3-epoch training run
=======================================
Quick test to verify training pipeline works end-to-end.

Usage:
    python scripts/smoke_test.py --approach a
    python scripts/smoke_test.py --approach b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess

def main():
    for approach in ['a', 'b']:
        print(f"\n{'='*60}")
        print(f"SMOKE TEST: Approach {approach.upper()}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable, 'scripts/train.py',
            '--approach', approach,
            '--smoke_test',
            '--batch_size', '4',
            '--d_model', '256',
            '--n_encoder_layers', '2',
            '--n_decoder_layers', '2',
            '--n_heads', '4',
            '--epochs', '3',
            '--save_every', '1',
            '--device', 'auto',
        ]
        
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            print(f"\n❌ Approach {approach.upper()} FAILED")
            return 1
        else:
            print(f"\n✅ Approach {approach.upper()} PASSED")
    
    print(f"\n{'='*60}")
    print("ALL SMOKE TESTS PASSED!")
    print(f"{'='*60}")
    return 0

if __name__ == '__main__':
    exit(main())
