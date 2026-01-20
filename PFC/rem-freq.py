import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 500.0
ARQUIVO = "EMG500_1.csv"

x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]

# Rejeita-faixa (bandstop) de 60 a 61 Hz
LOW = 60.0
HIGH = 62.0
ORDER = 4

sos = butter(ORDER, [LOW, HIGH], btype="bandstop", fs=FS, output="sos")
y = sosfiltfilt(sos, x)

t = np.arange(len(x)) / FS

plt.figure(figsize=(12,4))
#plt.plot(t, x, label="Antes", linewidth=0.8)
plt.plot(t, y, label="Depois (bandstop 60–61 Hz)", linewidth=0.8)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.title("EOG - antes/depois do rejeita-faixa 60–61 Hz")
plt.grid(True)
plt.legend()
plt.xlim(0, 60)
plt.tight_layout()
plt.show()


