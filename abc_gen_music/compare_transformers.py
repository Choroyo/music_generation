import json
import matplotlib.pyplot as plt

# Carga los datos
paths = {
    "600": "abc_transformer_model600/checkpoint-36140/trainer_state.json",
    "1200": "abc_transformer_model1200/checkpoint-37155/trainer_state.json",
    "3000": "abc_transformer_model3000/checkpoint-37680/trainer_state.json"
}

metrics = {}

for key, path in paths.items():
    with open(path, 'r') as f:
        data = json.load(f)
        losses = [entry['loss'] for entry in data['log_history'] if 'loss' in entry]
        epochs = [entry['epoch'] for entry in data['log_history'] if 'epoch' in entry]
        metrics[key] = (epochs, losses)

# Graficar
plt.figure(figsize=(10, 6))
for label, (epochs, losses) in metrics.items():
    plt.plot(epochs, losses, label=f"Model {label}")

plt.title("Comparación de pérdida por modelo")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("transformers_comparison.png")
print(" Plot saved as transformers_comparison.png")