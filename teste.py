import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 1. Carregar os dados
df = pd.read_csv("vendas_filtradas_quantum.csv", sep=";", encoding="latin1")
df.rename(columns={"data_dia": "ds", "total_venda_dia_kg": "y"}, inplace=True)
df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%d")

# 2. Dividir os dados em treino (80%) e teste (20%)
tamanho = len(df)
tamanho_treino = int(tamanho * 0.8)

df_treino = df.iloc[:tamanho_treino]
df_teste = df.iloc[tamanho_treino:]

# 3. Criar e treinar o modelo Prophet
modelo = Prophet()
modelo.fit(df_treino)

# 4. Criar datas para previsão com base no conjunto de teste
datas_teste = df_teste["ds"]
futuro = pd.DataFrame({"ds": datas_teste})

# 5. Gerar previsão para o período de teste
previsao = modelo.predict(futuro)

# 6. Avaliação do modelo
y_true = df_teste["y"].values
y_pred = previsao["yhat"].values

rmse = mean_squared_error(y_true, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# 7. Exibir gráfico
fig = plot_plotly(modelo, modelo.predict(modelo.make_future_dataframe(periods=0)))
fig.show()

# 8. (Opcional) Comparar real vs previsto
comparacao = pd.DataFrame({
    "Data": datas_teste,
    "Real (kg)": y_true,
    "Previsto (kg)": y_pred
})
print(comparacao.head())

# 9. Exportar previsão para CSV
comparacao.to_csv("avaliacao_prevista_vs_real.csv", index=False)
