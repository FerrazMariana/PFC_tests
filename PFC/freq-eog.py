import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 1000.0
ARQUIVO = "piscada_simples.csv"

# Banda típica pra EOG/piscada (ajuste se quiser)
F_LOW  = 0.2   # Hz  (remove drift muito lento)
F_HIGH = 15  # Hz  (remove EMG/ruído mais rápido)
ORDER = 6

# carrega 1 coluna do CSV (sem pandas)
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]

# filtro passa-faixa (mantém só a banda do EOG)
sos = butter(ORDER, [F_LOW, F_HIGH], btype="bandpass", fs=FS, output="sos")
y = sosfiltfilt(sos, x)

# plota
t = np.arange(len(x)) / FS
plt.figure(figsize=(12,4))
plt.plot(t, x, label="Bruto", linewidth=0.8)
plt.plot(t, y, label=f"Filtrado {F_LOW}-{F_HIGH} Hz", linewidth=1.0)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.title("EOG - passa-faixa")
plt.grid(True)
plt.legend()
plt.xlim(0, 60)  # zoom (mude como quiser)
plt.tight_layout()
plt.show()
