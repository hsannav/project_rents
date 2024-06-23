import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import pickle
import json

from xgboost import XGBRegressor
from sklearn.inspection import PartialDependenceDisplay

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

with open("barrios.geojson", "r") as f:
    barrios_geojson = json.load(f)

with open('modelos.pkl', 'rb') as file:
    modelos = pickle.load(file)

st.set_page_config(layout="wide")
st.title('Alquileres en Valencia')

pisos = pd.read_excel('pisos.xlsx')
pisos['Barrio'] = [barrio.upper() for barrio in pisos['Barrio']]

pisos.loc[pisos['Barrio'] == 'BARRIO DE FAVARA', 'Barrio'] = 'FAVARA'
pisos.loc[pisos['Barrio'] == 'BENIMÀMET', 'Barrio'] = 'BENIMAMET'
pisos.loc[pisos['Barrio'] == 'BETERÓ', 'Barrio'] = 'BETERO'
pisos.loc[pisos['Barrio'] == 'CAMÍ DE VERA', 'Barrio'] = 'CAMI DE VERA'
pisos.loc[pisos['Barrio'] == 'CAMÍ FONDO', 'Barrio'] = 'CAMI FONDO'
pisos.loc[pisos['Barrio'] == 'CAMÍ REIAL', 'Barrio'] = 'CAMI REAL'
pisos.loc[pisos['Barrio'] == 'CASTELLAR-OLIVERAL', 'Barrio'] = "CASTELLAR-L'OLIVERAL"
pisos.loc[pisos['Barrio'] == 'CIUTAT JARDÍ', 'Barrio'] = 'CIUTAT JARDI'
pisos.loc[pisos['Barrio'] == 'CIUTAT UNIVERSITÀRIA', 'Barrio'] = 'CIUTAT UNIVERSITARIA'
pisos.loc[pisos['Barrio'] == 'EL BOTÀNIC', 'Barrio'] = 'EL BOTANIC'
pisos.loc[pisos['Barrio'] == 'EL CABANYAL-EL CANYAMELAR', 'Barrio'] = 'CABANYAL-CANYAMELAR'
pisos.loc[pisos['Barrio'] == 'EXPOSICIÓ', 'Barrio'] = 'EXPOSICIO'
pisos.loc[pisos['Barrio'] == 'GRAN VÍA', 'Barrio'] = 'LA GRAN VIA'
pisos.loc[pisos['Barrio'] == 'MISLATA', 'Barrio'] = 'SOTERNES'
pisos.loc[pisos['Barrio'] == 'MONT-OLIVET', 'Barrio'] = 'MONTOLIVET'
pisos.loc[pisos['Barrio'] == 'NOU BENICALAP', 'Barrio'] = 'BENICALAP'
pisos.loc[pisos['Barrio'] == 'PLAYA DE LA MALVARROSA', 'Barrio'] = 'LA MALVA-ROSA'
pisos.loc[pisos['Barrio'] == 'SANT LLORENÇ', 'Barrio'] = 'SANT LLORENS'
pisos.loc[pisos['Barrio'] == 'SANT MARCELLÍ', 'Barrio'] = 'SANT MARCEL.LI'
pisos.loc[pisos['Barrio'] == 'XIRIVELLA', 'Barrio'] = 'LA LLUM'

pisos = pisos.loc[pisos['Planta'] <= 9, :]

precios_medios = pisos.groupby("Barrio")["Precio"].mean().reset_index()

barrios_gdf = gpd.GeoDataFrame.from_features(barrios_geojson["features"]).merge(
    precios_medios, left_on="nombre", right_on="Barrio", how="left"
)

fig_mapa = px.choropleth_mapbox(
    barrios_gdf,
    geojson=barrios_geojson,
    locations="Barrio",
    featureidkey="properties.nombre",
    color="Precio",
    color_continuous_scale="YlOrRd",
    mapbox_style="carto-positron",
    zoom=11,
    center={"lat": 39.45, "lon": -0.37739},
    opacity=0.7,
    labels={"Precio": "Precio medio del alquiler (€)"},
    hover_name="Barrio",
    hover_data={"Barrio": False, "Precio": True},
)
fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, clickmode='event+select')

pisos['Precio'] = [float(precio) for precio in pisos['Precio']]
X = pisos.loc[:, ['Habitaciones', 'Planta', 'Superficie', 'Exterior', 'Ascensor', 'Precio']]
X = X.dropna()
y = X['Precio']
X = X.loc[:, ['Habitaciones', 'Planta', 'Superficie', 'Exterior', 'Ascensor']]

features = ['Planta', 'Exterior', 'Ascensor', 'Habitaciones', 'Superficie']

def crear_pdp(barrio):
    if barrio == 'TODOS LOS BARRIOS':
        modelo = modelos[barrio]
        title = f"Influencia de las características en el precio de la vivienda en Valencia (sobre {len(pisos['Barrio'])} muestras)"
    else:
        modelo = modelos[barrio]
        title = f"Influencia de las características en el precio de la vivienda en {barrio.title()} (sobre {sum(pisos['Barrio'] == barrio)} muestras)"
    pdp = PartialDependenceDisplay.from_estimator(modelo, X, features)
    plt.close()
    fig_pdp1 = make_subplots(rows=1, cols=3, subplot_titles=features[:3])
    fig_pdp2 = make_subplots(rows=1, cols=2, subplot_titles=features[3:])

    for i, feature in enumerate(features):
        values = pdp.pd_results[i]['values'][0]
        values = values[~np.isnan(values)]
        average = [round(avg, 2) for avg in pdp.pd_results[i]['average'][0]]
        pdp_df = pd.DataFrame({'values': values, 'average': average})
        if 0 in pdp_df['values']:
            pdp_df['values'] = pdp_df['values'].replace({0: "No", 1: "Sí"})

        if i < 3:
            fig_pdp1.add_trace(go.Scatter(
                x=pdp_df["values"],  
                y=pdp_df["average"],    
                name=feature,
                line=dict(color=f'rgb({i*50}, {255-i*50}, 0)')  
            ), row=1, col=i+1)
            fig_pdp1.update_traces(hovertemplate="Los pisos con esta característica se alquilan en promedio por %{y}€<extra></extra>")
        else:
            fig_pdp2.add_trace(go.Scatter(
                x=pdp_df["values"],  
                y=pdp_df["average"],    
                name=feature,
                line=dict(color=f'rgb({i*50}, {255-i*50}, 0)')  
            ), row=1, col=i%3+1)
            fig_pdp2.update_traces(hovertemplate="Los pisos con esta característica se alquilan en promedio por %{y}€<extra></extra>")

    fig_pdp1.update_layout(
        title=title,
        showlegend=False,
        height=400, width=1200,
    )
    fig_pdp2.update_layout(
        title='',
        showlegend=False,
        height=400, width=1200,
    )
    st.plotly_chart(fig_pdp1)
    st.plotly_chart(fig_pdp2)

barrios = ["TODOS LOS BARRIOS"] + pisos['Barrio'].unique().tolist()

col1, col2 = st.columns([0.50, 0.50])

with st.container():
    
    with col1:
        st.plotly_chart(fig_mapa)
        barrio_seleccionado = st.selectbox('Selecciona un barrio:', barrios)
        col11, col12, col13, col14 = st.columns([0.25, 0.25, 0.25, 0.25])

        with col11:
            habitaciones = st.selectbox('Habitaciones:', [i for i in range(2, 6)])

        with col12:
            planta = st.selectbox('Planta del piso:', [i for i in range(2, 9)])

        with col13:
            exterior = st.selectbox('Exterior:', ['Sí', 'No'])

        with col14:
            ascensor = st.selectbox('Ascensor:', ['Sí', 'No'])
        
        superficie = st.slider('Superficie', min_value=50, max_value=150, value=100)

        if barrio_seleccionado == 'TODOS LOS BARRIOS':
            barrio_pred = 'Valencia'
        else:
            barrio_pred = barrio_seleccionado

        pred = modelos[barrio_seleccionado].predict(np.array([habitaciones, planta, superficie, int(exterior == 'Sí'), int(ascensor == 'Sí')]).reshape(1, -1))

        st.markdown(f'El alquiler de un piso de estas características en **{barrio_pred.title()}** costaría **{pred[0]:.2f}€** al mes')

    with col2:
        crear_pdp(barrio_seleccionado)

st.markdown('Creado por Hugo Sánchez y María Verdú en junio de 2024')
