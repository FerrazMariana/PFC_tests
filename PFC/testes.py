import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 500.0
ARQUIVO = "ECG500_1.csv"   # troque aqui
COLUNA = 0

# Corte superior (high-frequency filter)
F_HIGH = 30.0

# Dois cortes inferiores para comparar
F_LOW_1 = 0.2   # Hz (típico p/ preservar mais conteúdo lento)
F_LOW_2 = 3.0   # Hz (remove artefato de suor/deriva, como no enunciado)

ORDER = 4

# ---- Leitura ----
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, COLUNA]
x = x[np.isfinite(x)]
x = x - np.mean(x)

t = np.arange(len(x)) / FS

def bandpass(sig, fs, f_low, f_high, order=4):
    sos = butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)

y_05_70 = bandpass(x, FS, F_LOW_1, F_HIGH, ORDER)
y_3_70  = bandpass(x, FS, F_LOW_2, F_HIGH, ORDER)

# ---- Plot ----
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
#plt.plot(t, x, linewidth=0.7, label="Bruto")
plt.plot(t, y_05_70, linewidth=0.9, label=f"Filtrado {F_LOW_1}-{F_HIGH} Hz")
plt.grid(True)
plt.ylabel("Sinal")
plt.legend()

plt.subplot(2, 1, 2)
#plt.plot(t, x, linewidth=0.7, label="Bruto")
plt.plot(t, y_3_70, linewidth=0.9, label=f"Filtrado {F_LOW_2}-{F_HIGH} Hz (low=3 Hz)")
plt.grid(True)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.legend()

plt.xlim(0, 60)  # ajuste o zoom como quiser
plt.tight_layout()
plt.show()
