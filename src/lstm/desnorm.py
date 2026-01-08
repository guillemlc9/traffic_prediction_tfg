"""
desnorm.py
----------
Script per desnormalitzar les prediccions i valors reals del model LSTM
per fer-los comparables amb els resultats d'ARIMA.

Aquest script:
1. Carrega els paràmetres de normalització (mean, std)
2. Carrega y_test i y_pred del model LSTM
3. Desnormalitza els valors
4. Calcula les mètriques MAE, MSE i RMSE en l'escala original
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict

# Afegim el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))


def load_normalization_params(params_path: str) -> Dict[str, float]:
    """
    Carrega els paràmetres de normalització (mean, std).
    
    Args:
        params_path: Ruta al fitxer JSON amb els paràmetres
    
    Returns:
        Diccionari amb 'mean' i 'std'
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    print(f"Paràmetres de normalització carregats:")
    print(f"  Mean: {params['mean']:.4f}")
    print(f"  Std:  {params['std']:.4f}\n")
    
    return params


def denormalize(normalized_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Desnormalitza els valors aplicant la transformació inversa de z-score.
    
    Formula: x_original = (x_normalized * std) + mean
    
    Args:
        normalized_values: Array amb valors normalitzats
        mean: Mitjana utilitzada en la normalització
        std: Desviació estàndard utilitzada en la normalització
    
    Returns:
        Array amb valors desnormalitzats
    """
    return (normalized_values * std) + mean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula les mètriques MAE, MSE i RMSE.
    
    Args:
        y_true: Valors reals
        y_pred: Valors predits
    
    Returns:
        Diccionari amb les mètriques
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }


def main():
    """
    Script principal per desnormalitzar prediccions LSTM.
    """
    
    # Rutes dels fitxers
    # PD: Segons el model que es vulgui desnormalitzar, es canvia el directori
    MODEL_DIR = "models/lstm/20251227_143405_lr1e-03_bs1024_u64_w36_h1"
    NORM_PARAMS_PATH = "models/lstm/normalization_params.json"
    
    model_dir = Path(MODEL_DIR)
    
    # 1. Carreguem paràmetres de normalització
    params = load_normalization_params(NORM_PARAMS_PATH)
    mean = params["mean"]
    std = params["std"]
    
    # 2. Carreguem y_test i y_pred (normalitzats)
    y_test_norm = np.load(model_dir / "y_test.npy")
    y_pred_norm = np.load(model_dir / "y_pred_test.npy")
    
    print(f"   Shape y_test: {y_test_norm.shape}")
    print(f"   Shape y_pred: {y_pred_norm.shape}\n")
    
    # 3. Desnormalitzem valors
    y_test_denorm = denormalize(y_test_norm, mean, std)
    y_pred_denorm = denormalize(y_pred_norm, mean, std)
    
    print(f"    y_test desnormalitzat")
    print(f"    y_pred desnormalitzat\n")
    
    # 4. Calculem mètriques normalitzades
    metrics_norm = compute_metrics(y_test_norm, y_pred_norm)
    metrics_denorm = compute_metrics(y_test_denorm, y_pred_denorm)

    print(f"MAE:  {metrics_norm['mae']:.6f}")
    print(f"MSE:  {metrics_norm['mse']:.6f}")
    print(f"RMSE: {metrics_norm['rmse']:.6f}")
    
    print(f"MAE:  {metrics_denorm['mae']:.6f}")
    print(f"MSE:  {metrics_denorm['mse']:.6f}")
    print(f"RMSE: {metrics_denorm['rmse']:.6f}")
    
    # 5. Guardem resultats desnormalitzats
    output_dir = model_dir / "denormalized"
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "y_test_denorm.npy", y_test_denorm)
    np.save(output_dir / "y_pred_denorm.npy", y_pred_denorm)
    
    # Guardem mètriques desnormalitzades
    metrics_output = {
        "normalized": metrics_norm,
        "denormalized": metrics_denorm,
        "normalization_params": {
            "mean": mean,
            "std": std
        }
    }
    
    with open(output_dir / "metrics_denorm.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"    y_test_denorm.npy guardat")
    print(f"    y_pred_denorm.npy guardat")
    print(f"    metrics_denorm.json guardat")
    print(f"Resultats guardats a: {output_dir}")
    
    # 6. Mostrem exemple de valors
    print("\n" + "=" * 60)
    print("EXEMPLE DE VALORS (primeres 10 mostres)")
    print("=" * 60)
    print(f"{'Index':<8} {'y_test_norm':<15} {'y_test_denorm':<15} {'y_pred_norm':<15} {'y_pred_denorm':<15}")
    print("-" * 68)
    for i in range(min(10, len(y_test_norm))):
        print(f"{i:<8} {y_test_norm[i]:<15.4f} {y_test_denorm[i]:<15.4f} {y_pred_norm[i]:<15.4f} {y_pred_denorm[i]:<15.4f}")
    print("=" * 60)
    
    print("\nProcés completat!")
    print("\nAra pots comparar les mètriques desnormalitzades amb les d'ARIMA.")


if __name__ == "__main__":
    main()
