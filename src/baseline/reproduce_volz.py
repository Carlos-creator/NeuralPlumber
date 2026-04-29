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

from models.dcgan import DCGAN_G
from metrics.structural import structural_score

MODEL_PATH = DAGSTUHL_PYTORCH / "netG_epoch_5000.pth"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "baseline"


def load_generator(model_path: Path, nz: int = 32, nc: int = 10, ngf: int = 64) -> DCGAN_G:
    gen = DCGAN_G(32, nz, nc, ngf, 0, 0)
    gen.load_state_dict(torch.load(str(model_path), map_location="cpu"))
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


def main(n_samples: int = 100):
    print(f"Cargando modelo desde: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"ERROR: No se encontró {MODEL_PATH}")
        print("Asegúrate de que el repositorio DagstuhlGAN esté en ../clone/DagstuhlGAN/")
        return

    gen = load_generator(MODEL_PATH)
    print(f"Generando {n_samples} niveles...")
    levels = sample_levels(gen, n=n_samples)

    print("\n=== Métricas estructurales baseline (Volz et al., n={}) ===".format(n_samples))
    metrics = compute_metrics(levels)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    # Guardar resultados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "metrics_baseline.json"
    with open(out_file, "w") as f:
        json.dump({"n_samples": n_samples, "metrics": metrics}, f, indent=2)
    print(f"\nResultados guardados en: {out_file}")

    # Guardar algunos niveles de ejemplo
    sample_file = OUTPUT_DIR / "sample_levels.json"
    with open(sample_file, "w") as f:
        json.dump([l.tolist() for l in levels[:10]], f)
    print(f"10 niveles de ejemplo guardados en: {sample_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    main(args.n_samples)
