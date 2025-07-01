import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from pandasgui import show

# ========== PARTE 1: FILTRO E TRATAMENTO ==========

df = pd.read_csv("./data/dados_vendas_sinteticos.csv", encoding="latin1", sep=";")

# show(df)

df["data_dia"] = pd.to_datetime(df["data_dia"], format="%d/%m/%Y")

data_inicio = "2025-03-25"
data_fim = "2025-06-22"

df_retirada = df[
    (df["data_dia"] >= data_inicio)
    & (df["data_dia"] <= data_fim)
    & (df["id_produto"] == 237497)
]

# show(df_retirada)

df_retirada.to_csv("vendas_filtradas_quantum.csv", index=False, encoding="latin1", sep=";")

df_retirada.info()

primeiras_informacoes = df_retirada.head()

# show(primeiras_informacoes)

qtd_total_kg_dia = df_retirada["total_venda_dia_kg"]

df["qtd_total_venda_mes"] = (df_retirada["total_venda_dia_kg"]).sum()

df_total = pd.DataFrame({
    "qtd_total_venda_mes": df["qtd_total_venda_mes"]
})

# show(df_total)

# show(qtd_total_venda_mes)

# achar as principais médias:
df_retirada["media"] = df_retirada["total_venda_dia_kg"].mean()

df_media = pd.DataFrame({
    "media_total": df_retirada["media"]
})

# show(df_retirada)
# show(df_media)

# ver a questão da taxa de retração
# taxa de retração de 5%
porcentagem = 0.05
df_retirada["peso_perdido"] = df["total_venda_dia_kg"] * porcentagem

# show(df_retirada)

# saber o quanto tenho que colocar para descongelar
df_retirada["colocar_descongelar_kg"] = df_retirada["total_venda_dia_kg"] + df_retirada["peso_perdido"]

show(df_retirada)

# peso_retirada = df["peso_util"] / 1 - taxa_retracao;

# tratar com nossas métricas MASP e RM

# criar o script para automatizar a criação dos relatórios, e o modelo preditivo:

# ========== PARTE 2: MODELO PREDITIVO (PROPHET) ==========

df = pd.read_csv("vendas_filtradas_quantum.csv", sep=';')

# Renomear colunas para padrão do Prophet
df.rename(columns={'data_dia': 'ds', 'total_venda_dia_kg': 'y'}, inplace=True)

df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

df.shape()

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

# separar uma parte dos dados para teste
# 20%
lenght_to_test = len(df) * 0.2

# separar uma parte dos dados para treino
# 80%
lenght_to_training = len(df) * 0.8

df["treino"] = df["ds"][:250]
df["treino"] = df["y"][:250]

df["teste"] = df["ds"][50:]
df["teste"] = df["y"][50:]
