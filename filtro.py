
import pandas as pd;

from pandasgui import show;

df = pd.read_csv("./data/dados_vendas_sinteticos.csv",encoding="latin1", sep=";");

#show(df);

df["data_dia"] = pd.to_datetime(df["data_dia"], format="%d/%m/%Y");

data_inicio = "2025-03-25";

data_fim = "2025-06-22";

df_retirada = df[(df["data_dia"] >= data_inicio ) & (df["data_dia"] <= data_fim) & (df["id_produto"]==237497)];

#show(df_retirada);

df_retirada.to_csv("vendas_filtradas_quantum.csv",index = False,encoding="latin1", sep=";");

df_retirada.info();

primeiras_informacoes = df_retirada.head();

#show(primeiras_informacoes);

qtd_total_kg_dia = df_retirada["total_venda_dia_kg"];

df["qtd_total_venda_mes"] = (df_retirada["total_venda_dia_kg"]).sum();

df_total = pd.DataFrame({
    "qtd_total_venda_mes": df["qtd_total_venda_mes"]
})

#show(df_total);

#show(qtd_total_venda_mes);

#achar as principais medias: 

df_retirada["media"] = df_retirada['total_venda_dia_kg'].mean();

df_media = pd.DataFrame({
    "media_total" : df_retirada["media"]
})

#show(df_retirada);

#show(df_media);

#ver a questao da taxa de retracao
#taxa de retracao de 5%

porcentagem = 0.05;

df_retirada["peso_perdido"] = df["total_venda_dia_kg"] * porcentagem;

#show(df_retirada);

#que coisas consigo fazer com essa taxa de retracao?
#saber o quanto tenho que colocar para descongelar 

df_retirada["colocar_descongelar_kg"] = df_retirada["total_venda_dia_kg"] + df_retirada["peso_perdido"];

show(df_retirada);

#peso_retirada = df["peso_util"] / 1 - taxa_retracao;


#tratar com nossas metricas MASP E RM

#criar o script para automatizar a criacao dos relatorios


