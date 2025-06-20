import pandas as pd # type: ignore
from scipy.optimize import curve_fit # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
sns.set_theme()

tabela = pd.read_csv("posicoes.csv")
tabela = tabela[500:]
tabela["x"] -= (np.max(tabela["x"]) + np.min(tabela["x"])) / 2 
tabela["x"] *= 2 / ((np.max(tabela["x"]) - np.min(tabela["x"])))
tabela["x"] *= (36.5/2)/100

def f(t, A, w, phi, b):
    return A * np.exp(-b * t) * np.cos(w * t - phi)

parametros_iniciais = [0.1, 2*np.pi, 0, 0.1]
parametros, _ = curve_fit(
    f, tabela["t"], tabela["x"],
    p0=parametros_iniciais,
    bounds=([0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]),
    maxfev=5000
)

tabela["fit"] = f(tabela["t"], *parametros)

sns.scatterplot(data=tabela, x="t", y="x")
sns.lineplot(
    data=tabela, x="t", y="fit",
    label=f"Fit: A={parametros[0]:.2f}, w={parametros[1]:.2f}, phi={parametros[2]:.2f}, b={parametros[3]:.2f}",
    color="red"
)
plt.title("x(t)")
plt.show()

periodo = (2*np.pi) / parametros[1]
fatorDeQualidade = 2*np.pi / (1 - np.exp(-2*parametros[3]*periodo))

print("Fator de qualidade:", fatorDeQualidade)
print("Periodo:", periodo)
