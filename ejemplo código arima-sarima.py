import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import plotly.express as px
from matplotlib.pyplot import xlabel


df = pd.read_csv('Indicadores.csv')
df = df.sort_values(by='Periodo').reset_index(drop=True)

px.line(x = df["Periodo"], y = df["Valor"] , labels = {"x":"Periodo","y":"Valor"} , title="ÍNDICE GLOBAL DE ACTIVIDAD ECONÓMICA (MÉXICO)"")

df['ln(Valor)'] = np.log(df['Valor'])

print(adfuller(df['ln(Valor)'])[1])

df['First_diff']= df['ln(Valor)']  - np.log(df['Valor']).shift(1)
df['Anual_diff']= df['ln(Valor)']  - np.log(df['Valor']).shift(12)
df

print(adfuller(df['Anual_diff'].dropna())[1])


plot = plot_acf(df['Anual_diff'].dropna(), lags = 12)

plot = plot_pacf(df['Anual_diff'].dropna(), lags = 12)


d = 0 
D = 1
P = 0
Q = 0
p = 1
q = 0
modelo = sm.tsa.statespace.SARIMAX(df['ln(Valor)'], order=(p,d,q), seasonal_order = (P,D,Q,12), trend='c',  simple_differencing=True)

modelo_fit = modelo.fit(disp = 0)
print(modelo_fit.summary())

plot = modelo_fit.plot_diagnostics(figsize=(10,10))

residuos = modelo_fit.resid

plot = plot_acf(residuos, lags = 12)

plot = plot_pacf(residuos, lags = 12)



import numpy as np
import math
modelo = sm.tsa.statespace.SARIMAX(df['ln(Valor)'], order=(p,d,q), seasonal_order = (P,D,Q,12), trend='c', simple_differencing=True)
modelo_fit = modelo.fit()
predicciones  = np.asarray((modelo_fit.forecast(steps=24)))
forecast= pd.DataFrame()
forecast["Periodo"] = np.asarray(["2022/08","2022/09","2022/10","2022/11",
                       "2022/12","2023/01","2023/02","2023/03",
                       "2023/04","2023/05","2023/06","2023/07",
                       "2023/08","2023/09","2023/10","2023/11",
                       "2023/12","2024/01","2024/02","2024/03",
                       "2024/04","2024/05","2024/06","2024/07"])
forecast["Predict"] = predicciones

forecast["Predict_exp"] = forecast["Predict"] ** math.e

#fusionamos ambos dataframes
final_df = pd.concat([df, forecast])
final_df = final_df.set_index("Periodo", drop = True)
final_df = final_df[["Valor", "Predict"]]

final_df[200:].plot()
