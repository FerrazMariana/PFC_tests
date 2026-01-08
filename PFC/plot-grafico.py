import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fs = 1000  # Hz
arquivo = "piscada_dupla_forte.csv"

x = pd.read_csv(arquivo, header=None).iloc[:, 0].astype(float).to_numpy()
t = np.arange(len(x)) / fs

plt.plot(t, x)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.title("EOG")
plt.grid(True)
plt.show()


