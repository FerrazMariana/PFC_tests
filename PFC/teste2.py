import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 1000
ARQUIVO = "EMG1000_1.csv"  # seu CSV (1 coluna)

# Rejeita-faixa (rede elétrica)
NOTCH_LOW  = 60.0
NOTCH_HIGH = 62.0
ORDER_NOTCH = 4

# Filtros como na figura
LP = 15      # passa-baixa 70 Hz
HP_A = 0.5     # passa-alta 0.5 Hz
HP_B = 6     # passa-alta 3 Hz
ORDER_HP = 4
ORDER_LP = 4

# Plot (zoom opcional no tempo e na FFT)
T0 = 0
T1 = 60        # None para tudo
F_MAX = 200   # Hz (limite do gráfico de FFT)

def bandstop(sig, fs, low, high, order=4):
    sos = butter(order, [low, high], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)

def hp_then_lp(sig, fs, hp, lp, order_hp=4, order_lp=4):
    sos_hp = butter(order_hp, hp, btype="highpass", fs=fs, output="sos")
    y = sosfiltfilt(sos_hp, sig)
    sos_lp = butter(order_lp, lp, btype="lowpass", fs=fs, output="sos")
    y = sosfiltfilt(sos_lp, y)
    return y

def fft_mag(sig, fs):
    n = len(sig)
    w = np.hanning(n)
    X = np.fft.rfft(sig * w)
    f = np.fft.rfftfreq(n, d=1/fs)
    mag = (2.0 / np.sum(w)) * np.abs(X)  # normalização para comparação visual
    return f, mag

# --------- Leitura ----------
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]
x = x - np.mean(x)

t = np.arange(len(x)) / FS
if T1 is not None:
    m = (t >= T0) & (t <= T1)
else:
    m = slice(None)

# --------- Notch ----------
x_notch = bandstop(x, FS, NOTCH_LOW, NOTCH_HIGH, ORDER_NOTCH)

# --------- Filtragens (a) e (b) ----------
y_a = hp_then_lp(x_notch, FS, HP_A, LP, ORDER_HP, ORDER_LP)  # HP 0.5 / LP 70
y_b = hp_then_lp(x_notch, FS, HP_B, LP, ORDER_HP, ORDER_LP)  # HP 3   / LP 70

# --------- FFTs ----------
f_x, mag_x = fft_mag(x, FS)
f_n, mag_n = fft_mag(x_notch, FS)
f_a, mag_a = fft_mag(y_a, FS)
f_b, mag_b = fft_mag(y_b, FS)

# --------- Plot ----------
plt.figure(figsize=(12, 9))

# Tempo (a)
plt.subplot(3, 1, 1)
plt.plot(t[m], y_a[m], linewidth=0.9)
plt.title(f"(a) Bandstop 60–61 Hz + HP={HP_A} Hz + LP={LP} Hz")
plt.ylabel("Sinal")
plt.grid(True)

# Tempo (b)
plt.subplot(3, 1, 2)
plt.plot(t[m], y_b[m], linewidth=0.9)
plt.title(f"(b) Bandstop 60–61 Hz + HP={HP_B} Hz + LP={LP} Hz")
plt.ylabel("Sinal")
plt.grid(True)

# FFT (comparações)
plt.subplot(3, 1, 3)
plt.plot(f_x, mag_x, linewidth=0.8, label="FFT Original")
plt.plot(f_n, mag_n, linewidth=0.8, label="FFT Após bandstop 60–61")
plt.plot(f_a, mag_a, linewidth=0.9, label=f"FFT (a) HP {HP_A} / LP {LP}")
plt.plot(f_b, mag_b, linewidth=0.9, label=f"FFT (b) HP {HP_B} / LP {LP}")
plt.xlim(0, min(F_MAX, FS/2))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT (magnitude) - comparação antes/depois")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
