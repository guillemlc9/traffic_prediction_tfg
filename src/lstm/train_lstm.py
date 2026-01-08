"""
train_lstm.py
-------------
Mòdul per entrenar el model LSTM on:
1. Carreguem les dades .npy
2. Definim l'arquitectura del model: 1 capa de 32 o 64 unitats, una Dense intermitja de 32 i una capa
   final Dense de 1 unitat.
3. Configurar entrenament: loss = mse, optimizer = adam (1e-3), metrics = mae
4. Entrenar el model per 100 epochs
5. Guardar el model
"""

import sys
from pathlib import Path
import json
import platform
from typing import Any, Dict

# Afegir el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from datetime import datetime
from src.lstm.create_sequences import load_sequences


def build_lstm_model(
    window_size: int = 36,
    lstm_units: int = 64,
    dense_units: int = 32,
    learning_rate: float = 3e-4
) -> keras.Model:
    """
    Construeix el model LSTM.
    
    Args:
        window_size: Nombre de timesteps a la finestra d'entrada
        lstm_units: Nombre d'unitats a la capa LSTM (32 o 64)
        dense_units: Nombre d'unitats a la capa Dense intermèdia
        learning_rate: Learning rate per l'optimitzador Adam
    
    Returns:
        Model LSTM compilat
    """
    model = models.Sequential([
        # Capa LSTM
        layers.LSTM(
            units=lstm_units,
            input_shape=(window_size, 1),
            name="lstm_layer"
        ),
        
        # Capa Dense intermèdia
        layers.Dense(
            units=dense_units,
            activation="relu",
            name="dense_intermediate"
        ),
        
        # Capa de sortida
        layers.Dense(
            units=1,
            activation="linear",
            name="output"
        )
    ])
    
    # Compilar el model
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    
    return model


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 1024,
    run_dir: Path = "models/lstm"
) -> keras.callbacks.History:
    """
    Entrena el model LSTM amb callbacks.
    
    Args:
        model: Model LSTM a entrenar
        X_train: Dades d'entrenament
        y_train: Targets d'entrenament
        X_val: Dades de validació
        y_val: Targets de validació
        epochs: Nombre d'epochs
        batch_size: Mida del batch
        model_dir: Directori on guardar els checkpoints
    
    Returns:
        Historial d'entrenament
    """
    ckpt_path = run_dir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callback_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=str(ckpt_path / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard
        callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=0
        )
    ]
    
    # Entrenar el model
    print("\n" + "="*60)
    print("ENTRENAMENT DEL MODEL LSTM")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        shuffle=True,
        verbose=1
    )
    
    return history


def save_final_model(
    model: keras.Model,
    run_dir: Path
):
    """
    Guarda el model final entrenat.
    
    Args:
        model: Model a guardar
        model_dir: Directori on guardar el model
    """
    
    filepath = run_dir / "final.keras"
    model.save(filepath)
    print(f"\nModel final guardat a: {filepath}")


def print_model_summary(model: keras.Model):
    """
    Mostra un resum del model.
    """
    print("\n" + "="*60)
    print("ARQUITECTURA DEL MODEL")
    print("="*60)
    model.summary()
    print("="*60 + "\n")


def make_run_dir(base_dir: str, config: Dict[str, Any]) -> Path:
    """
    Crea un directori únic per l'execució, per no sobreescriure res.
    Ex: models/lstm/20251227_133004_lr5e-03_bs1024_u64_w36_h1
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr = config["learning_rate"]
    bs = config["batch_size"]
    u = config["lstm_units"]
    w = config["window_size"]
    h = config["horizon"]
    run_name = f"{ts}_lr{lr:.0e}_bs{bs}_u{u}_w{w}_h{h}"
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, obj: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def save_training_history(history: keras.callbacks.History, run_dir: Path):
    history_clean = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    save_json(run_dir / "training_history.json", history_clean)


def save_predictions_and_metrics(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history: keras.callbacks.History,
    run_dir: Path,
    batch_size: int
):
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).squeeze()

    # Guardar arrays
    np.save(run_dir / "y_test.npy", y_test)
    np.save(run_dir / "y_pred_test.npy", y_pred)

    # Mètriques
    test_metrics = compute_metrics(y_test, y_pred)

    metrics_payload = {
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_mae": float(min(history.history["val_mae"])),
        "test": test_metrics
    }

    save_json(run_dir / "metrics.json", metrics_payload)
    print(f"metrics.json guardat a: {run_dir / 'metrics.json'}")
    

def main():
    """
    Script principal per entrenar el model LSTM.
    """

    print("Dispositius disponibles per TensorFlow:")
    print(tf.config.list_physical_devices())

    BASE_DIR = "models/lstm"
    SEQUENCES_DIR = "data/processed/lstm_sequences"
    WINDOW_SIZE = 36
    HORIZON = 1
    LSTM_UNITS = 64
    DENSE_UNITS = 32
    LEARNING_RATE = 3e-4
    EPOCHS = 50
    BATCH_SIZE = 1024

    config = {
        "SEQUENCES_DIR": SEQUENCES_DIR,
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "lstm_units": LSTM_UNITS,
        "dense_units": DENSE_UNITS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "legacy.Adam",
        "loss": "mse",
        "metric": "mae"
    }

    run_dir = make_run_dir(BASE_DIR, config)
    print(f"\nRun dir: {run_dir}")

    # info d'entorn (molt útil per la memòria)
    env_info = {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "devices": [d.name for d in tf.config.list_physical_devices()],
        "platform": platform.platform()
    }
    save_json(run_dir / "config.json", {**config, **env_info})
    
    print("="*60)
    print("ENTRENAMENT MODEL LSTM - PREDICCIÓ DE TRÀNSIT")
    print("="*60)
    
    # 1. Carregar seqüències
    print(f"\n1. Carregant seqüències de {SEQUENCES_DIR}...")
    sequences = load_sequences(SEQUENCES_DIR)
    
    X_train = sequences["X_train"]
    y_train = sequences["y_train"]
    X_val = sequences["X_val"]
    y_val = sequences["y_val"]
    X_test = sequences["X_test"]
    y_test = sequences["y_test"]
    
    print(f"   ✓ Train: {X_train.shape}, {y_train.shape}")
    print(f"   ✓ Val:   {X_val.shape}, {y_val.shape}")
    print(f"   ✓ Test:  {X_test.shape}, {y_test.shape}")
    
    # 2. Construir model
    print(f"\n2. Construint model LSTM...")
    print(f"   - LSTM units: {LSTM_UNITS}")
    print(f"   - Dense units: {DENSE_UNITS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    
    model = build_lstm_model(
        window_size=WINDOW_SIZE,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        learning_rate=LEARNING_RATE
    )
    
    print_model_summary(model)
    
    # 3. Entrenar model
    print(f"\n3. Entrenant model per {EPOCHS} epochs...")
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        run_dir=run_dir
    )
    
    # 4. Avaluar en test
    print("\n4. Avaluant en conjunt de test...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss (MSE): {test_loss:.4f}")
    print(f"   Test MAE: {test_mae:.4f}")

    # Guardar history
    save_training_history(history, run_dir)

    # Guardar prediccions + metrics
    best_model = tf.keras.models.load_model(run_dir / "checkpoints" / "best.keras")
    save_predictions_and_metrics(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        history=history,
        run_dir=run_dir,
        batch_size=BATCH_SIZE
    )

    best_model.save(run_dir / "best.keras")
    
    # 5. Guardar model final
    save_final_model(model, run_dir)
    
    # Resum final
    print("\n" + "="*60)
    print("ENTRENAMENT COMPLETAT!")
    print("="*60)
    print(f"Millor val_loss: {min(history.history['val_loss']):.4f}")
    print(f"Millor val_mae: {min(history.history['val_mae']):.4f}")
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Model guardat a: {run_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
