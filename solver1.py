import numpy as np
import pandas as pd
from scipy.optimize import minimize

coeficientes = pd.DataFrame({
    'B': [0, -2, -4, 0, -1, 0],
    'C': [0, -3, 0, -1, 0, 0],
    'D': [0, 0, -2, 0, 0, 4],
    'E': [16, 18, 4, 2, 1, 0],
    'F': [1, 0, 2, 0, 0, -1],
    'G': [0, 2, 1, 0, 0, 0],
    'H': [4, 7, 3, 1, 0, 0],
    'ind': [-6, -12, -6, 0, 0, 0]
}, index=['C:', 'H:', 'O:', 'N:', 'S:', 'RQ:'])

inicio = np.array([1, 1, 1, 1, 1, 1, 1])  

def objective(variables):
    vars_series = pd.Series(variables, index=['B', 'C', 'D', 'E', 'F', 'G', 'H'])
    resultados = coeficientes[['B', 'C', 'D', 'E', 'F', 'G', 'H']].dot(vars_series) + coeficientes['ind']
    cuadrados = resultados**2
    F_O = cuadrados.sum()
    return F_O

limites = [(0.1, None)]

resultado = minimize(objective, inicio, method='SLSQP', bounds=limites)

print('Estado de optimización:', resultado.success)
print('Mensaje:', resultado.message)
print('Valor óptimo de la función objetivo:', resultado.fun)
print('Valores óptimos de las variables:', resultado.x)

variables_opt = pd.Series(resultado.x, index=['B', 'C', 'D', 'E', 'F', 'G', 'H'])
print(variables_opt)

resultados_opt = coeficientes[['B', 'C', 'D', 'E', 'F', 'G', 'H']].dot(variables_opt) + coeficientes['ind']
cuadrados_opt = resultados_opt**2

output_df = pd.DataFrame({
    'resultado': resultados_opt,
    'cuadrados': cuadrados_opt
})

output_df.loc['F.O.'] = [resultado.fun, np.nan]

print(output_df)
print()
print(f'Valor de F: {resultado.x[4]}')
print(f'Valor de D: {resultado.x[2]}')
print(resultado.x[4]/resultado.x[2])