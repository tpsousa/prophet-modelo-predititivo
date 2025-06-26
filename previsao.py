import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly

df = pd.read_csv("vendas_filtradas_quantum.csv", sep=';')

# Renomear colunas para padrão do Prophet
df.rename(columns={'data_dia': 'ds', 'total_venda_dia_kg': 'y'}, inplace=True)

df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

np.random.seed(4587)

# Criar e treinar o modelo
m = Prophet()
m.fit(df)

# Criar datas futuras
futuro = m.make_future_dataframe(periods=365, freq='D')

# Gerar previsões
previsao = m.predict(futuro)

# Plotar resultado
fig = plot_plotly(m, previsao)
fig.show()
