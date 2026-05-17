"""
Reproduce los experimentos de Volz et al. (2018) y mide métricas baseline.
Genera 100 niveles aleatorios con el modelo preentrenado y reporta
las métricas estructurales del paper base como punto de comparación.

Requisito: netG_epoch_5000.pth en ../../clone/DagstuhlGAN/pytorch/
"""
import sys
import os
import torch
import numpy as np
import json
from pathlib import Path

# Path al repositorio base
DAGSTUHL_PYTORCH = Path(__file__).resolve().parents[3] / "clone" / "DagstuhlGAN" / "pytorch"
sys.path.insert(0, str(DAGSTUHL_PYTORCH))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dcgan import DCGAN_G, load_compatible
from metrics.structural import structural_score

MODEL_PATH = DAGSTUHL_PYTORCH / "netG_epoch_5000.pth"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "baseline"


def load_generator(model_path: Path, nz: int = 32, nc: int = 10, ngf: int = 64) -> DCGAN_G:
    gen = DCGAN_G(32, nz, nc, ngf, 0, 0)
    load_compatible(gen, str(model_path))
    gen.eval()
    return gen


def generate_level(generator: DCGAN_G, z: np.ndarray = None) -> np.ndarray:
    """Genera un nivel desde un vector latente (aleatorio si z=None)."""
    if z is None:
        z_tensor = torch.randn(1, 32, 1, 1)
    else:
        z_tensor = torch.FloatTensor(z).view(1, 32, 1, 1)
    with torch.no_grad():
        out = generator(z_tensor)                 # (1, 10, 32, 32)
    level = out[0, :, :14, :28].argmax(dim=0)    # (14, 28)
    return level.numpy()


def sample_levels(generator: DCGAN_G, n: int = 100) -> list:
    return [generate_level(generator) for _ in range(n)]


def compute_metrics(levels: list) -> dict:
    all_m = [structural_score(np.array(l)) for l in levels]
    keys  = all_m[0].keys()
    return {k: float(np.mean([m[k] for m in all_m])) for k in keys}


def main(n_samples: int = 100, model_path: Path = MODEL_PATH, output_dir: Path = OUTPUT_DIR):
    print(f"Cargando modelo desde: {model_path}")
    if not model_path.exists():
        print(f"ERROR: No se encontró {model_path}")
        return

    gen = load_generator(model_path)
    print(f"Generando {n_samples} niveles...")
    levels = sample_levels(gen, n=n_samples)

    print("\n=== Métricas estructurales (n={}) ===".format(n_samples))
    metrics = compute_metrics(levels)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "metrics_baseline.json"
    with open(out_file, "w") as f:
        json.dump({"n_samples": n_samples, "model": str(model_path), "metrics": metrics}, f, indent=2)
    print(f"\nResultados guardados en: {out_file}")

    sample_file = output_dir / "sample_levels.json"
    with open(sample_file, "w") as f:
        json.dump([l.tolist() for l in levels[:10]], f)
    print(f"10 niveles de ejemplo guardados en: {sample_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",   type=int, default=100)
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    args = parser.parse_args()
    main(args.n_samples, Path(args.model_path), Path(args.output_dir))
