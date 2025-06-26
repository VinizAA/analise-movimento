import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import base64, io, time, random
from typing import List
from mini_neuralnetwork import predict, extract, answer
from streamlit_browser_storage import LocalStorage
from st_login_form import login_form, logout
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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

def response_generator(ai_answer):
    for word in ai_answer.split():
        yield word + " "
        time.sleep(0.09)

@st.dialog("Deseja realmente sair?", width="small")
def my_logout():
    if st.button("Sim", type="primary"):
        st.success("Deslogando...")
        time.sleep(0.5)
        logout()

def main_app(user_id, username):
    st.set_page_config(page_title="Visualizador de Movimento", layout="wide")
    data = None

    with st.sidebar:
        logo_big = "logo_big.png"
        logo_small = "logo_small.png"
        st.logo(logo_big, size="large", icon_image=logo_small)

        if st.button("Logout", icon=":material/logout:", use_container_width=True):
            my_logout()

        uploaded_file = st.file_uploader("Escolha um arquivo .csv", type="csv")
        if uploaded_file is None:
            st.info("Adicione um arquivo .csv para começar!", icon="☝️")
        else:
            data = pd.read_csv(uploaded_file)

        if data is not None:
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            chat = st.container(height=570)

            for msg in st.session_state.chat_messages:
                chat.chat_message(msg["role"]).write(msg["content"])

            prompt = st.chat_input("Digite sua pergunta")
            if prompt:
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                chat.chat_message("user").write(prompt)

                resposta = answer(data, prompt)
                if isinstance(resposta, dict):
                    response_text = resposta.get("text", "")
                else:
                    response_text = str(resposta)

                chat.chat_message("assistant").write_stream(response_generator(response_text))
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

    if data is not None: 
        tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "🕸️ Grafo", "🧠 Análise DatAI"])

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

            ord_jnts = ["shoulderL", "shoulderR", "elbowL", "elbowR", "kneeL", "kneeR"]
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
                ord_opt_user = [jnt for jnt in ord_jnts if jnt in opt_user]
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
            chosen_time = st.slider("Escolha o tempo (s)",
                                    min_value=0.0,
                                    max_value=((data['time'].max()) - (data['time'].min())) / 40,
                                    value=0.0,
                                    step=0.1,
                                    format="%.2f")
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

                q3 = data[joint].quantile(2 / 3)
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

        with tab3:          
            if data is not None and len(data) > 0:                
                nomes = {
                    "shoulderLangle": "Ombro Esquerdo",
                    "shoulderRangle": "Ombro Direito",
                    "elbowLangle": "Cotovelo Esquerdo",
                    "elbowRangle": "Cotovelo Direito",
                    "kneeLangle": "Joelho Esquerdo",
                    "kneeRangle": "Joelho Direito"
                }

                available_cols = [col for col in nomes if col in data.columns]
                escolha = st.selectbox("Selecione a articulação:", [nomes[k] for k in available_cols], key="resumo_art")

                col_map = {v: k for k, v in nomes.items()}
                coluna = col_map.get(escolha)

                st.subheader(f"📌 Estatísticas de {escolha}")
                col1, col2 = st.columns(2)
                if coluna in data.columns:
                    with col1:
                        st.metric("Valor Máximo", f"{data[coluna].max():.1f}°")
                        st.metric("Valor Mínimo", f"{data[coluna].min():.1f}°")
                    with col2:
                        st.metric("Média", f"{data[coluna].mean():.1f}°")
                        st.metric("Amplitude", f"{(data[coluna].max() - data[coluna].min()):.1f}°")
                else:
                    st.warning("Articulação não encontrada nos dados.")

                st.line_chart(data[coluna])

                st.divider()
                st.markdown("### 📊 Análise por Articulação")
                analysis_data = []

                for joint_key, joint_name in nomes.items():
                    if joint_key in data.columns:
                        series = data[joint_key]
                        mean_angle = series.mean()
                        std_angle = series.std()
                        min_angle = series.min()
                        max_angle = series.max()
                        range_angle = max_angle - min_angle
                        avg_movement = series.diff().abs().mean()

                        if avg_movement > 2.0:
                            movement_class = "Alto"
                            movement_color = "🔴"
                        elif avg_movement > 1.0:
                            movement_class = "Médio"
                            movement_color = "🟡"
                        else:
                            movement_class = "Baixo"
                            movement_color = "🟢"

                        analysis_data.append({
                            "Articulação": joint_name,
                            "Ângulo Médio": f"{mean_angle:.1f}°",
                            "Variação (±)": f"{std_angle:.1f}°",
                            "Amplitude": f"{range_angle:.1f}°",
                            "Movimento": f"{movement_color} {movement_class}",
                            "Min": f"{min_angle:.1f}°",
                            "Max": f"{max_angle:.1f}°"
                        })

                df_analysis = pd.DataFrame(analysis_data)
                st.dataframe(df_analysis, use_container_width=True, hide_index=True)


                st.divider()
                st.markdown("### 📈 Perfil de Movimento")
                if len(available_cols) >= 3:
                    import plotly.graph_objects as go

                    categories = []
                    values = []

                    for joint_key in available_cols:
                        categories.append(nomes[joint_key])
                        movement_score = min(100, (data[joint_key].std() / 50) * 100)
                        values.append(movement_score)

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Intensidade de Movimento',
                        line_color='rgb(0,123,255)'
                    ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        height=500,
                        title="Perfil de Intensidade por Articulação"
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Carregue um arquivo CSV para ver as análises automáticas dos dados de movimento.")
            
st.set_page_config(page_title="Visualizador de Movimento", layout="wide")

msg_placeholder_login = st.empty() 

if not st.session_state.get("authenticated", False):
    st.set_page_config(page_title="Bem-vindo!", layout="centered")
    supabase_connection = login_form()
else:
    user_id = st.session_state.get("user_id", "guest")
    username = st.session_state.get("username", None)

    if not st.session_state.get("welcome_shown", False):
        msg_placeholder_login.empty()
        if username:
            with st.spinner(f"Preparando tudo para você, {username}", show_time=False):
                time.sleep(random.randint(1,2))
            with st.spinner(f"Carregando DatAI de {username}", show_time=False):
                time.sleep(random.randint(1,2))
            with st.spinner(f"Entrando em sua conta, {username}", show_time=False):
                time.sleep(random.randint(1,2))
        else:
            with st.spinner("Preparando tudo..."):
                time.sleep(random.randint(1,2))
            with st.spinner("Carregando DatAI..."):
                time.sleep(random.randint(1,2))
            with st.spinner("Entrando..."):
                time.sleep(random.randint(1,2))

        time.sleep(0.5)
        st.session_state["welcome_shown"] = True

    msg_placeholder_login.empty()
    main_app(user_id, username)
