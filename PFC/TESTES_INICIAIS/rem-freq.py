import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 500.0
ARQUIVO = "data/piscada_dupla_500_diferencial_1.csv"

# -------- Limites de outlier (ajuste) --------
LIM_INF = -0.002
LIM_SUP =  0.001

def remove_outliers_interp(sig, lim_inf, lim_sup):
    """Marca outliers (sig<lim_inf ou sig>lim_sup) como NaN e interpola linearmente."""
    out = (sig < lim_inf) | (sig > lim_sup)
    sig2 = sig.copy()
    sig2[out] = np.nan

    idx = np.arange(len(sig2))
    good = np.isfinite(sig2)

    if np.sum(good) >= 2:
        sig2[~good] = np.interp(idx[~good], idx[good], sig2[good])
    else:
        sig2 = sig.copy()  # caso extremo

    return sig2, out

# -------- Leitura --------
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]

# remove DC
x = x - np.mean(x)

# -------- Remove outliers por interpolação --------
x, out_mask = remove_outliers_interp(x, LIM_INF, LIM_SUP)
print(f"Outliers interpolados: {np.sum(out_mask)} / {len(out_mask)} ({100*np.mean(out_mask):.2f}%)")

# -------- Bandstop 60–62 Hz --------
LOW = 60.0
HIGH = 62.0
ORDER = 4

sos = butter(ORDER, [LOW, HIGH], btype="bandstop", fs=FS, output="sos")
y = sosfiltfilt(sos, x)

# remove DC após filtrar
y = y - np.mean(y)

# -------- Tempo --------
t = np.arange(len(x)) / FS

plt.figure(figsize=(12,4))
plt.plot(t, x, label="Antes (sem outliers interp.)", linewidth=0.8)
plt.plot(t, y, label=f"Depois (bandstop {LOW:.0f}–{HIGH:.0f} Hz)", linewidth=0.8)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.title("EOG - antes/depois (com outliers removidos por interpolação)")
plt.grid(True)
plt.legend()
plt.xlim(0, 60)
plt.tight_layout()
plt.show()

# -------- FFT (antes e depois) --------
N = len(x)
X = np.fft.rfft(x)
Y = np.fft.rfft(y)
freq = np.fft.rfftfreq(N, d=1.0/FS)

mag_x = np.abs(X)  # sem normalização
mag_y = np.abs(Y)  # sem normalização

plt.figure(figsize=(12,4))
plt.plot(freq, mag_x, label="FFT Antes", linewidth=0.9)
plt.plot(freq, mag_y, label="FFT Depois", linewidth=0.9)
plt.xlabel("Frequência (Hz)")
plt.ylabel("|X(f)|")
plt.title("FFT - antes/depois do bandstop (com outliers interp.)")
plt.grid(True)
plt.legend()
plt.xlim(0, 200)
plt.tight_layout()
plt.show()
