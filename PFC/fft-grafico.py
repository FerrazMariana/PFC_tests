import numpy as np
import matplotlib.pyplot as plt

FS = 1000.0  # Hz
ARQUIVO = "piscada_dupla_forte.csv"

# --- carrega 1 coluna numérica do CSV (sem pandas) ---
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:  # se vier com mais de uma coluna, pega a primeira
    x = x[:, 0]
x = x[np.isfinite(x)]

# remove média (DC) para a FFT ficar mais útil
x = x - np.mean(x)

# --- FFT (apenas metade positiva) ---
N = len(x)
X = np.fft.rfft(x)
freq = np.fft.rfftfreq(N, d=1.0/FS)

# magnitude (amplitude) normalizada
mag = np.abs(X) / N

# --- plota no domínio do tempo e frequência ---
t = np.arange(N) / FS

plt.figure(figsize=(12, 4))
plt.plot(t, x)
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("Sinal no tempo (média removida)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(freq, mag)
plt.xlabel("Frequência (Hz)")
plt.ylabel("|X(f)| (normalizado)")
plt.title("FFT (magnitude)")
plt.grid(True)
plt.xlim(0, 100)  # EOG costuma estar bem abaixo disso; ajuste se quiser
plt.tight_layout()
plt.show()
