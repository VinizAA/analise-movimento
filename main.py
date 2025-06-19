import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
from typing import List
import plotly.graph_objects as go

def break_in_3(data: pd.DataFrame, coluna: str):
    min_val = data[coluna].min()
    max_val = data[coluna].max()
    step = (max_val - min_val) / 3

    return [
        (min_val, min_val + step),
        (min_val + step, min_val + 2 * step),
        (min_val + 2 * step, max_val)
    ]

def plot_graphic(data: pd.DataFrame, jnts: List[str], options: dict):
    fig = go.Figure() 

    x_axis = data['time'] / 40
    ini_time = x_axis.min()
    x_axis_format = x_axis - ini_time

    for jnt in jnts:
        label = options[jnt]
        y_axis = data[f'{jnt}angle']
        fig.add_trace(go.Scatter(x=x_axis_format, y=y_axis, mode='lines', name=label, line_shape='spline'))

    fig.update_layout(
        xaxis=dict(title="Tempo (s)"),
        yaxis=dict(title="Ângulo (graus)"),
        template="plotly_dark", 
        hovermode="x unified", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.write("#### Gráfico das articulações selecionadas:")
    st.plotly_chart(fig, use_container_width=True)

def plot_graph(data: pd.DataFrame, time_sec: float = 1.0):
    time_format = data['time'] - data['time'].min()
    x_axis_format = ((time_format / 40) - time_sec).abs().idxmin()
     
    angles = {
        "shoulderLangle": data.at[x_axis_format, "shoulderLangle"],
        "shoulderRangle": data.at[x_axis_format, "shoulderRangle"],
        "elbowLangle": data.at[x_axis_format, "elbowLangle"],
        "elbowRangle": data.at[x_axis_format, "elbowRangle"],
        "kneeLangle": data.at[x_axis_format, "kneeLangle"],
        "kneeRangle": data.at[x_axis_format, "kneeRangle"]
    }

    id = {
        "shoulderLangle": "OE",
        "shoulderRangle": "OD",
        "elbowLangle": "CE",
        "elbowRangle": "CD",
        "kneeLangle": "JE",
        "kneeRangle": "JD"
    }

    positions = {
        "shoulderLangle": (-1, 1),
        "shoulderRangle": (1, 1),
        "elbowLangle": (-2, 0),
        "elbowRangle": (2, 0),
        "kneeLangle": (-1, -1),
        "kneeRangle": (1, -1),
    }

    edges = [
        ("shoulderLangle", "elbowLangle"),
        ("shoulderRangle", "elbowRangle"),
        ("shoulderLangle", "shoulderRangle"),
        ("shoulderLangle", "kneeLangle"),
        ("shoulderRangle", "kneeRangle"),
    ]

    x_left, y_left = positions["shoulderLangle"]
    x_right, y_right = positions["shoulderRangle"]

    x_middle = (x_left + x_right) / 2
    y_middle = ((y_left + y_right) / 2) - 1

    mid_key = "midShoulders"
    positions[mid_key] = (x_middle, y_middle)

    edges += [
        ("kneeLangle", mid_key),
        ("kneeRangle", mid_key)
    ]

    fig = go.Figure()
     
    line_color = "#9CA3AF"
    bg_color = "rgba(0,0,0,0)" 
    
    #arestas 
    for a, b in edges:
        x0, y0 = positions[a]
        x1, y1 = positions[b]

        if (a == "shoulderLangle" and b == "kneeLangle") or (a == "shoulderRangle" and b == "kneeRangle"):
            continue 
        else:
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="none"
            ))

    #arestas joelhos
    for joint in ["kneeLangle", "kneeRangle"]:
        x0, y0 = positions[joint]
        x1, y1 = positions["midShoulders"]

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=line_color, width=2),
            hoverinfo="none"
        ))

    #arestas corpo
    x0, y0 = 0, 0
    x1, y1 = 0, 1

    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode="lines",
        line=dict(color=line_color, width=2),
        hoverinfo="none"
    ))

    text_id = "#F9FAFB"    
    text_color = "#111827" 

    #nós
    for key, (x, y) in positions.items():
        if key in angles:
            angle = angles[key]

            intervals = break_in_3(data, key)

            if intervals [0][0] <= angle < intervals [0][1]:
                node_color = "green" 
            elif intervals [1][0] <= angle < intervals [1][1]:
                node_color = "yellow" 
            elif intervals [2][0] <= angle <= intervals [2][1]:
                node_color = "red"
            else:
                node_color = "blue"

            border_color = node_color

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=40, color=node_color, line=dict(color=border_color, width=2)),
                text=[f"{angle:.1f}°"],
                textposition="middle center",
                hoverinfo="text"
            )) 

            if key == "kneeLangle":
                x_id = x - 0.05
            elif key == "kneeRangle":
                x_id = x + 0.05
            else:
                x_id = x

            fig.add_trace(go.Scatter(
                x=[x_id], y=[y + 0.25], 
                mode="text",
                text=[id[key]],
                textposition="bottom center",
                showlegend=False,
                hoverinfo="none",
                textfont=dict(color=text_id, size=14, family="Arial")
            ))

    #figura
    fig.update_layout(
        title=f"Time = {time_sec:.2f}s",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)



st.set_page_config(page_title="Visualizador de Movimento", layout="wide")
st.title("📊 Visualizador de Dados de Movimento")

uploaded_file = st.file_uploader("Escolha um arquivo .csv", type="csv")

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        tab1, tab2 = st.tabs(["📈 Gráfico", "🕸️ Grafo"])

        with tab1:
            options = {
                "shoulderL": "Ombro Esquerdo",
                "shoulderR": "Ombro Direito",
                "elbowL": "Cotovelo Esquerdo",
                "elbowR": "Cotovelo Direito",
                "kneeL": "Joelho Esquerdo",
                "kneeR": "Joelho Direito",
            }

            shortcuts = {
                "Membros Inferiores": {"kneeL", "kneeR"},
                "Membros Superiores": {"shoulderL", "shoulderR", "elbowL", "elbowR"},
                "Lado Esquerdo": {"shoulderL", "elbowL", "kneeL"},
                "Lado Direito": {"shoulderR", "elbowR", "kneeR"},
                "Todos": {"kneeL", "kneeR", "shoulderL", "shoulderR", "elbowL", "elbowR"}
            }

            ord_jnts = ["shoulderL", "shoulderR", "elbowL", "elbowR", "kneeL", "kneeR"] #correct order of joints
            opt_user = set()

            st.write("### Selecione as articulações para visualização:")

            col_sh, col_el, col_kn, col_sets, col_sides = st.columns(5)

            with col_sh:
                st.markdown("**Ombros**")
                if st.checkbox(options["shoulderL"], key="shoulderL_"): 
                    opt_user.add("shoulderL")
                if st.checkbox(options["shoulderR"], key="shoulderR_"):
                    opt_user.add("shoulderR")

            with col_el:
                st.markdown("**Cotovelos**")
                if st.checkbox(options["elbowL"], key="elbowL_"):
                    opt_user.add("elbowL")
                if st.checkbox(options["elbowR"], key="elbowR_"):
                    opt_user.add("elbowR")

            with col_kn:
                st.markdown("**Joelhos**")
                if st.checkbox(options["kneeL"], key="kneeL_"):
                    opt_user.add("kneeL")
                if st.checkbox(options["kneeR"], key="kneeR_"):
                    opt_user.add("kneeR")

            with col_sets:
                st.markdown("**Grupos**")
                if st.checkbox("Todos", key="Todos_"):
                    opt_user.update(shortcuts["Todos"])
                if st.checkbox("Membros Superiores", key="Membros Superiores_"):
                    opt_user.update(shortcuts["Membros Superiores"])
                if st.checkbox("Membros Inferiores", key="Membros Inferiores_"):
                    opt_user.update(shortcuts["Membros Inferiores"])

            with col_sides:
                st.markdown("**Lados**")
                if st.checkbox("Lado Esquerdo", key="Lado Esquerdo_"):
                    opt_user.update(shortcuts["Lado Esquerdo"])
                if st.checkbox("Lado Direito", key="Lado Direito_"):
                    opt_user.update(shortcuts["Lado Direito"])

            if opt_user:
                ord_opt_user = [jnt for jnt in ord_jnts if jnt in opt_user] #['kneeL', 'kneeR']
                plot_graphic(data, ord_opt_user, options)
            else:
                st.info("❗ Por favor, selecione ao menos uma articulação para visualização.")

            with st.expander("🔍 Ver dados brutos"):
                cols = list(data.columns)
                if 'r_ankleZ' in cols:
                    idx = cols.index('r_ankleZ') + 1  
                    data = data.iloc[:, :idx]

                st.dataframe(data)

        with tab2:
            chosen_time = st.slider("Escolha o tempo (s)",min_value=0.0,max_value=((data['time'].max()) - (data['time'].min())) / 40,value=0.0, step=0.1, format="%.2f")
            plot_graph(data, chosen_time)
        
            with st.expander("Tempos com ângulos extremos"):
                nomes = {
                    "shoulderLangle": "Ombro Esquerdo",
                    "shoulderRangle": "Ombro Direito",
                    "elbowLangle": "Cotovelo Esquerdo",
                    "elbowRangle": "Cotovelo Direito",
                    "kneeLangle": "Joelho Esquerdo",
                    "kneeRangle": "Joelho Direito"
                }

                escolha_nome = st.selectbox("Escolha a articulação para exibir:", list(nomes.values()))

                joint = None
                for key, nome in nomes.items():
                    if nome == escolha_nome:
                        joint = key
                        break

                q3 = data[joint].quantile(2/3)
                max_val = data[joint].max()

                df_sel = data.loc[(data[joint] >= q3) & (data[joint] <= max_val), ['time', joint]].copy()

                df_sel['time_sec'] = ((df_sel['time'] - data['time'].min()) / 40).round(2)
                df_sel['angle'] = df_sel[joint].round(1)

                df_sel = df_sel.sort_values(by='angle', ascending=False).drop_duplicates(subset='time_sec')
                df_sel.sort_values(by='time_sec', inplace=True)

                if not df_sel.empty:
                    st.markdown(f"{len(df_sel)} ocorrência(s) com ângulo elevado!")
                    for _, row in df_sel.iterrows():
                        st.markdown(f"- Em {row['time_sec']}s: ângulo de {row['angle']}°")
                else:
                    st.markdown("- Nenhum valor no 3º intervalo.")
                    
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.error("Certifique-se de que o arquivo CSV está no formato correto e que as colunas 'time' e as colunas de ângulo (ex: 'shoulderLangle') existem.")
