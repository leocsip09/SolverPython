import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/leonardoponzebellido/Documents/Tarea modelos matemáticos fase 2.xlsx', 
                   skiprows=3, usecols='A:C', 
                   names=['Muestra', 'Tiempo (h)', 'Concentración (mg/L)'])
df_crecimiento = df.iloc[6:45].copy()

def modelo_riccati(t, C, X0, mu):
    exp_term = np.exp(mu * t)
    denom = C * exp_term + 1 - C * X0
    return 1 / denom

def objective(params, t, X):
    C, X0, mu = params
    X_pred = modelo_riccati(t, C, X0, mu)
    return np.sum((X - X_pred) ** 2)

tiempo = df_crecimiento['Tiempo (h)'].values
concentracion = df_crecimiento['Concentración (mg/L)'].values

inicio = np.array([0.01, 100, 0.001])
limites = [(None, None), (1, None), (None, None)]

resultado = minimize(objective, inicio, args=(tiempo, concentracion), method='L-BFGS-B', bounds=limites)

print('Estado de optimización:', resultado.success)
print('Mensaje:', resultado.message)
print('Valor óptimo de la función objetivo:', resultado.fun)
print('Valores óptimos de los parámetros:', resultado.x)

C_opt, X0_opt, mu_opt = resultado.x
concentracion_ajustada = modelo_riccati(tiempo, C_opt, X0_opt, mu_opt)

df_crecimiento['Concentración ajustada (mg/L)'] = concentracion_ajustada

print(df_crecimiento)

plt.plot(tiempo, concentracion, 'bo', label='Datos experimentales')
plt.plot(tiempo, concentracion_ajustada, 'r-', label='Modelo ajustado')
plt.xlabel('Tiempo (h)')
plt.ylabel('Concentración (mg/L)')
plt.legend()
plt.title('Ajuste de la ecuación de Riccati al crecimiento bacteriano')
plt.show()

print(f'Valor de C: {C_opt}')
print(f'Valor de X0: {X0_opt}')
print(f'Valor de mu: {mu_opt}')
