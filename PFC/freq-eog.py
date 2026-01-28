import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 500.0
ARQUIVO = "data/piscada_dupla_500_diferencial_1.csv"

# Banda (ajuste se quiser)
F_LOW  = 0.3   # Hz
F_HIGH = 200  # Hz
ORDER_BANDPASS = 4

# Rejeita-faixa 59–61 Hz
NOTCH_LOW = 59.0
NOTCH_HIGH = 61.0
ORDER_NOTCH = 4

# carrega 1 coluna do CSV (sem pandas)
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]

LIM = 0.005
mask = (x >= -LIM) & (x <= LIM)
x = x[mask]

print(f"Removidos: {np.sum(~mask)} / {len(x)} ({100*np.mean(~mask):.2f}%)")

# 1) rejeita-faixa (remove 59–61 Hz)
sos_notch = butter(ORDER_NOTCH, [NOTCH_LOW, NOTCH_HIGH],
                   btype="bandstop", fs=FS, output="sos")
x2 = sosfiltfilt(sos_notch, x)

# 2) passa-faixa (mantém F_LOW–F_HIGH)
sos_bp = butter(ORDER_BANDPASS, [F_LOW, F_HIGH],
                btype="bandpass", fs=FS, output="sos")
y = sosfiltfilt(sos_bp, x2)

# plota
t = np.arange(len(x)) / FS
plt.figure(figsize=(12,4))
# plt.plot(t, x, label="Bruto", linewidth=0.8)
plt.plot(t, y, label=f"Bandstop 59–61 Hz + Bandpass {F_LOW}-{F_HIGH} Hz", linewidth=1.0)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.title("EOG - rejeita-faixa + passa-faixa")
plt.grid(True)
plt.legend()
plt.xlim(0, 60)
plt.tight_layout()
plt.show()

