import streamlit as st
import pandas as pd
import sqlite3

# Connessione al database
conn = sqlite3.connect("data/dati.db")
df = pd.read_sql("SELECT * FROM mobile", conn)
conn.close()

# Titolo dell'applicazione
st.title("Mobile Price-Dashboard")

# Colonne per le metriche
col1, col2, col3, col4 = st.columns(4)

col1.metric("Modelli", len(df))
col2.metric("Batteria media", round(df["battery_power"].mean(), 2))
col3.metric("RAM media (MB)", round(df["ram"].mean(), 2))
col4.metric("% con 4G", round(df["four_g"].mean() * 100, 2))

# Sottotitolo e Grafico


opzioni_di_prezzo = sorted(df["price_range"].unique())

scelta= st.sidebar.selectbox("Seleziona una fascia di prezzo:", opzioni_di_prezzo)

df_filtrato= df[df["price_range"]==scelta]

min_value= int(df_filtrato["ram"].min())
max_value= int(df_filtrato["ram"].max())

scelta2= st.sidebar.slider("RAM minima (MB)", min_value, max_value)

df_filtrato= df_filtrato[df_filtrato["ram"] >=scelta2]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Modelli", len(df_filtrato))
col2.metric("Batteria media", round(df_filtrato["battery_power"].mean(),2))
col3.metric("RAM media (MB)", round(df_filtrato["ram"].mean(),2))
col4.metric("% con 4G", round(df_filtrato["four_g"].mean()*100,2))


st.subheader("Distribuzione modelli filtrati")
conteggi = df_filtrato["price_range"].value_counts().sort_index()
st.bar_chart(conteggi)






