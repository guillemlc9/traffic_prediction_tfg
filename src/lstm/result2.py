"""
Script per visualitzar l'evolució de loss i val_loss durant l'entrenament del model LSTM.
Aquest gràfic permet:
- Justificar l'early stopping
- Mostrar estabilitat o overfitting del model
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history_path: str, output_path: str = None):
    """
    Genera un gràfic de loss i val_loss per epoch.
    
    Args:
        history_path: Ruta al fitxer training_history.json
        output_path: Ruta on guardar el gràfic (opcional)
    """
    # Carregar l'historial d'entrenament
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Extreure les mètriques
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    # Crear el gràfic
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, loss, 'b-', label='Training Loss (MSE)', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss (MSE)', linewidth=2)
    
    # Marcar el millor epoch (mínim val_loss)
    best_epoch = val_loss.index(min(val_loss)) + 1
    best_val_loss = min(val_loss)
    plt.axvline(x=best_epoch, color='green', linestyle='--', 
                label=f'Best Epoch ({best_epoch})', linewidth=1.5)
    plt.plot(best_epoch, best_val_loss, 'go', markersize=10)
    
    # Configurar el gràfic
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Evolució de Loss durant l\'Entrenament del Model LSTM\n(LR=3e-4)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Afegir anotació amb informació de l'early stopping
    total_epochs = len(loss)
    plt.text(0.02, 0.98, 
             f'Total epochs: {total_epochs}\nBest val_loss: {best_val_loss:.4f}\nBest epoch: {best_epoch}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    # Guardar o mostrar el gràfic
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gràfic guardat a: {output_path}")
    else:
        plt.show()
    
    # Imprimir estadístiques
    print("\n" + "="*60)
    print("ESTADÍSTIQUES D'ENTRENAMENT")
    print("="*60)
    print(f"Total d'epochs executades: {total_epochs}")
    print(f"Millor epoch: {best_epoch}")
    print(f"Best val_loss (MSE): {best_val_loss:.4f}")
    print(f"Final training loss: {loss[-1]:.4f}")
    print(f"Final validation loss: {val_loss[-1]:.4f}")
    
    # Analitzar overfitting
    if loss[-1] < val_loss[-1]:
        gap = val_loss[-1] - loss[-1]
        print(f"\nGap entre train i val loss: {gap:.4f}")
        if gap > 0.01:
            print("Possible overfitting detectat (val_loss > train_loss)")
        else:
            print("Model estable (gap mínim entre train i val)")
    
    print("="*60 + "\n")


def main():
    """Funció principal per generar el gràfic."""
    # Definir rutes
    project_root = Path(__file__).parent.parent.parent
    history_path = project_root / "models" / "lstm" / "20251229_131237_lr3e-04_bs1024_u64_w36_h1" / "training_history.json"
    output_path = project_root / "reports" / "lstm_training_history_lr3e-04.png"
    
    # Verificar que existeix el fitxer d'historial
    if not history_path.exists():
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {history_path}")
    
    # Generar el gràfic
    plot_training_history(str(history_path), str(output_path))


if __name__ == "__main__":
    main()
