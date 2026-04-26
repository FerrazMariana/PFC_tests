import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fs = 500  # Hz
arquivo = r"data\pedro\filtrado\piscada_dupla_1_pedro.csv"

df = pd.read_csv(arquivo)

print(df.head())       # para ver as colunas
print(df.columns)      # para confirmar os nomes

t = df.iloc[:, 0].astype(float).to_numpy()   # primeira coluna = tempo
x = df.iloc[:, 1].astype(float).to_numpy()   # segunda coluna = sinal

plt.plot(t, x)
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal")
plt.title("EOG")
plt.grid(True)
plt.show()


