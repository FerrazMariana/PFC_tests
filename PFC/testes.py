# -*- coding: utf-8 -*-
# EOG: diagnóstico (clipping), filtro (notch 60 + bandpass), e detecção de piscadas
# Requisitos: numpy, matplotlib, scipy
# Instalação: python -m pip install numpy matplotlib scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, find_peaks

FS = 1000.0  # Hz
ARQUIVO = "piscada_dupla_forte.csv"   # troque para o outro CSV se quiser

# ---- filtros (ajuste se necessário) ----
USE_NOTCH = True
NOTCH_F0 = 60.0
NOTCH_Q  = 60.0

BAND_LOW  = 0.3    # Hz
BAND_HIGH = 15.0   # Hz
BAND_ORDER = 4

# ---- detecção (ajuste se pegar pouco/muito) ----
REFRACTORY_MS = 250     # distância mínima entre piscadas (ms)
PROM_FACTOR   = 4.0     # mais alto = mais conservador (menos falsos)
DOUBLE_MIN_S  = 0.20
DOUBLE_MAX_S  = 0.60


def load_1col_csv(path):
    x = np.genfromtxt(path, delimiter=",", dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Sinal vazio ou muito curto.")
    return x

def robust_sigma(x):
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad if mad > 0 else np.std(x)

def notch(x, fs, f0=60.0, Q=60.0):
    b, a = iirnotch(w0=f0, Q=Q, fs=fs)
    return filtfilt(b, a, x)

def bandpass(x, fs, f1, f2, order=4):
    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def fft_mag(x, fs):
    x = x - np.mean(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1/fs)
    mag = np.abs(X) / len(x)
    return f, mag


# ------------------- MAIN -------------------
x = load_1col_csv(ARQUIVO)
t = np.arange(len(x)) / FS

# diagnóstico de clipping/saturação
xmin, xmax = np.min(x), np.max(x)
frac_min = np.mean(x == xmin) * 100
frac_max = np.mean(x == xmax) * 100

print(f"Arquivo: {ARQUIVO}")
print(f"Amostras: {len(x)} | Duração: {t[-1]:.3f} s | FS={FS} Hz")
print(f"Min: {xmin:.10g} (em {frac_min:.2f}% das amostras)")
print(f"Max: {xmax:.10g} (em {frac_max:.2f}% das amostras)")

# remover DC antes de filtrar
x0 = x - np.mean(x)

# filtros
xf = x0.copy()
if USE_NOTCH:
    xf = notch(xf, FS, f0=NOTCH_F0, Q=NOTCH_Q)
xf = bandpass(xf, FS, BAND_LOW, BAND_HIGH, order=BAND_ORDER)

# detector de piscadas (pode ser positiva ou negativa -> usa |xf|)
sig = np.abs(xf)
sigma = robust_sigma(sig)
prom = PROM_FACTOR * sigma
min_dist = int((REFRACTORY_MS/1000.0) * FS)

peaks, props = find_peaks(sig, prominence=prom, distance=min_dist)
blink_times = peaks / FS

# classificar piscada dupla por intervalo
double_pairs = []
i = 0
while i < len(blink_times) - 1:
    dt = blink_times[i+1] - blink_times[i]
    if DOUBLE_MIN_S <= dt <= DOUBLE_MAX_S:
        double_pairs.append((blink_times[i], blink_times[i+1], dt))
        i += 2
    else:
        i += 1

print("\nPiscadas detectadas (s):")
if len(blink_times) == 0:
    print("  (nenhuma com os parâmetros atuais)")
else:
    print("  " + ", ".join(f"{bt:.3f}" for bt in blink_times))

print("\nPiscadas duplas (t1, t2, dt):")
if not double_pairs:
    print("  (nenhuma pela regra atual)")
else:
    for a, b, dt in double_pairs:
        print(f"  {a:.3f}, {b:.3f}  dt={dt:.3f}s")

# --------- PLOTS ---------
# 1) Sinal bruto
plt.figure(figsize=(12,4))
plt.plot(t, x, linewidth=0.8)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal (bruto)")
plt.title("EOG bruto (sem filtro)")
plt.grid(True, alpha=0.3)
plt.xlim(0, 15)  # mude o zoom
plt.tight_layout()
plt.show()

# 2) Sinal filtrado + marcações
plt.figure(figsize=(12,4))
plt.plot(t, xf, linewidth=0.8, label="Filtrado (notch 60 + bandpass 0.5–15 Hz)")
if len(peaks) > 0:
    plt.scatter(blink_times, xf[peaks], s=18, label="Piscadas (picos)", zorder=3)
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("EOG filtrado + detecção de piscadas")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 10)  # mude o zoom
plt.tight_layout()
plt.show()

# 3) FFT antes/depois (para ver 60 Hz e a banda útil)
f1, m1 = fft_mag(x0, FS)
f2, m2 = fft_mag(xf, FS)

plt.figure(figsize=(12,4))
plt.plot(f1, m1, label="FFT antes (média removida)")
plt.plot(f2, m2, label="FFT depois (filtrado)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("|X(f)| (normalizado)")
plt.title("FFT antes/depois")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 150)
plt.tight_layout()
plt.show()

# 4) Zoom automático em torno das piscadas (se houver)
if len(blink_times) > 0:
    for k, bt in enumerate(blink_times[:5]):  # mostra até 5 primeiras
        c = int(bt * FS)
        w = int(0.8 * FS)  # janela ±0.8s
        a = max(0, c - w)
        b = min(len(xf), c + w)
        plt.figure(figsize=(10,3))
        plt.plot(t[a:b], xf[a:b], linewidth=0.9)
        plt.axvline(bt, linestyle="--", linewidth=1.0)
        plt.title(f"Zoom piscada #{k+1} em t={bt:.3f}s")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
