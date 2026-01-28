import numpy as np
import matplotlib.pyplot as plt

FS = 500.0  # Hz
ARQUIVO = "data/piscada_dupla_500_diferencial_1.csv"

# --- carrega 1 coluna numérica ---
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]

# remove média (DC)
x = x - np.mean(x)

# --- FFT (metade positiva) ---
N = len(x)
X = np.fft.rfft(x)
freq = np.fft.rfftfreq(N, d=1.0/FS)

# magnitude SEM normalização
mag = np.abs(X)

# --- plota apenas a FFT ---
plt.figure(figsize=(12, 4))
plt.plot(freq, mag)
plt.xlabel("Frequência (Hz)")
plt.ylabel("|X(f)|")
plt.title("FFT (magnitude) - sem normalização")
plt.grid(True)
plt.xlim(0, 200)
plt.tight_layout()
plt.show()
