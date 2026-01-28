import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FS = 500
ARQUIVO = "data/tests/ECG500.csv"  # seu CSV (1 coluna)

# ---------- Limites de outlier (ajuste) ----------
LIM_INF = -0.002
LIM_SUP =  0.001

# Rejeita-faixa (rede elétrica)
NOTCH_LOW  = 59.0
NOTCH_HIGH = 61.0
ORDER_NOTCH = 4

# Filtros como na figura
LP = 200
HP_A = 0.5
ORDER_HP = 4
ORDER_LP = 4

# Plot (zoom opcional no tempo e na FFT)
T0 = 0
T1 = 300
F_MAX = 200

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
    X = np.fft.rfft(sig)
    f = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(X)  # sem normalização e sem janela
    return f, mag


def remove_outliers_interp(sig, lim_inf, lim_sup):
    """
    Marca outliers (sig<lim_inf ou sig>lim_sup) como NaN e interpola linearmente.
    Mantém o mesmo tamanho do vetor.
    """
    out = (sig < lim_inf) | (sig > lim_sup)
    sig2 = sig.copy()
    sig2[out] = np.nan

    idx = np.arange(len(sig2))
    good = np.isfinite(sig2)

    # precisa de pelo menos 2 pontos bons para interpolar
    if np.sum(good) >= 2:
        sig2[~good] = np.interp(idx[~good], idx[good], sig2[good])
    else:
        sig2 = sig.copy()  # caso extremo

    return sig2, out

# --------- Leitura ----------
x = np.genfromtxt(ARQUIVO, delimiter=",", dtype=float)
if x.ndim > 1:
    x = x[:, 0]
x = x[np.isfinite(x)]
x = x - np.mean(x)

# --------- Remove outliers por interpolação ----------
x, out_mask = remove_outliers_interp(x, LIM_INF, LIM_SUP)
print(f"Outliers interpolados: {np.sum(out_mask)} / {len(out_mask)} ({100*np.mean(out_mask):.2f}%)")

t = np.arange(len(x)) / FS
m = (t >= T0) & (t <= T1) if T1 is not None else slice(None)

# --------- Notch ----------
x_notch = bandstop(x, FS, NOTCH_LOW, NOTCH_HIGH, ORDER_NOTCH)

# --------- Filtragens (a) e (b) ----------
y_a = hp_then_lp(x_notch, FS, HP_A, LP, ORDER_HP, ORDER_LP)

# --------- FFTs ----------
f_x, mag_x = fft_mag(x, FS)
f_n, mag_n = fft_mag(x_notch, FS)
f_a, mag_a = fft_mag(y_a, FS)

# --------- Plot ----------
plt.figure(figsize=(12, 9))

plt.subplot(2, 1, 1)
plt.plot(t[m], y_a[m], linewidth=0.9)
plt.title(f"(a) Bandstop {NOTCH_LOW:.0f}–{NOTCH_HIGH:.0f} Hz + HP={HP_A} Hz + LP={LP} Hz")
plt.ylabel("Sinal")
plt.grid(True)


plt.subplot(2, 1, 2)
plt.plot(f_x, mag_x, linewidth=0.8, label="FFT (após outlier interp.)")
plt.plot(f_n, mag_n, linewidth=0.8, label=f"FFT Após bandstop {NOTCH_LOW:.0f}–{NOTCH_HIGH:.0f}")
plt.plot(f_a, mag_a, linewidth=0.9, label=f"FFT (a) HP {HP_A} / LP {LP}")
plt.xlim(0, min(F_MAX, FS/2))
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT (magnitude) - comparação antes/depois")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --------- Salvar apenas tempo e sinal final (y_a) ----------
#OUT_CSV = "data/data_filtrado/d_p_d_500_1m_5_filtrado.csv"

#dados = np.column_stack([t, y_a])  # tempo, sinal final
#np.savetxt(OUT_CSV, dados, delimiter=",", header="t_s,sinal_filtrado", comments="")



