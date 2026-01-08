import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

FS = 1000.0
ARQUIVO = "piscada_dupla_forte.csv"

x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]

# Notch em 60 Hz
f0 = 60.0
Q = 60.0
b, a = iirnotch(f0, Q, fs=FS)
y = filtfilt(b, a, x)

t = np.arange(len(x)) / FS

plt.plot(t, x, label="Antes", linewidth=0.8)
plt.plot(t, y, label="Depois (notch 60 Hz)", linewidth=0.8)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.title("EOG - antes/depois do notch 60 Hz")
plt.grid(True)
plt.legend()
plt.xlim(0, 60)  # ajuste o zoom (aqui mostra só os 5 primeiros segundos)
plt.show()

