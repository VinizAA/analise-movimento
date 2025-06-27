import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import time, random

from datetime import datetime, date
from typing import List
from mini_neuralnetwork import predict, extract, answer
from streamlit_browser_storage import LocalStorage
from st_login_form import login_form, logout
from supabase import create_client

# 1. Config geral do app - só uma vez, no topo do arquivo
st.set_page_config(page_title="Datai App", page_icon=":material/home:", layout="centered")

# 2. Config Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

#Auxiliares
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
    import plotly.graph_objs as go

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

#páginas
def create_pacientes():
    st.set_page_config(page_title="Cadastro de Paciente", page_icon=":material/add:", layout="wide")
    st.markdown("# :material/add: Cadastro de pacientes")

    with st.form("form_paciente", clear_on_submit=True):
        nome = st.text_input("Nome")    
        sobrenome = st.text_input("Sobrenome")
        sexo = st.selectbox("Sexo", ["Masculino", "Feminino", "Outro"])
        date_str = st.text_input("Data de Nascimento (DD/MM/AAAA)")

        try:
            data_nascimento = datetime.strptime(date_str, "%d/%m/%Y").date()
        except:
            data_nascimento = None
            if date_str:
                st.warning("Data inválida! Use o formato DD/MM/AAAA.")

        foto = st.file_uploader("Foto do Paciente", type=["jpg", "jpeg", "png"])
        documento = st.file_uploader("Documento Médico", type=["pdf", "docx"])

        submitted = st.form_submit_button("Cadastrar")

        if submitted:
            if not all([nome, sobrenome, sexo, data_nascimento, foto, documento]):
                st.warning("Preencha todos os campos e envie os arquivos!")
                return

            hoje = date.today()
            idade = hoje.year - data_nascimento.year - ((hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day))

            with st.spinner("Enviando dados..."):
                foto_bytes = foto.read()
                foto_path = f"pacientes/fotos/{nome}_{sobrenome}_{int(time.time())}.{foto.name.split('.')[-1]}"
                supabase.storage.from_("pacientes").upload(foto_path, foto_bytes, {"content-type": foto.type})

                doc_bytes = documento.read()
                doc_path = f"pacientes/documentos/{nome}_{sobrenome}_{int(time.time())}.{documento.name.split('.')[-1]}"
                supabase.storage.from_("pacientes").upload(doc_path, doc_bytes, {"content-type": documento.type})

                data = {
                    "nome": nome,
                    "sobrenome": sobrenome,
                    "sexo": sexo,
                    "data_nascimento": data_nascimento.isoformat(),
                    "idade": idade,
                    "foto_url": foto_path,
                    "documento_url": doc_path
                }

                res = supabase.table("pacientes").insert(data).execute()
                if res.data:
                    st.success("Paciente cadastrado com sucesso!")
                else:
                    st.error("Erro ao cadastrar paciente.")

def patient(nome_completo):
    def inner():
        st.set_page_config(page_title=f"{nome_completo}", page_icon=":material/person:", layout="wide")
        st.title(f"Análise de movimento de **{nome_completo}**")

        data = None
        uploaded_file = st.file_uploader("Escolha um arquivo .csv", type="csv")
        if uploaded_file is None:
            st.info("Adicione um arquivo .csv para começar!", icon="☝️")
        else:
            data = pd.read_csv(uploaded_file)

        if data is not None: 
            tab1, tab2 = st.tabs(["🧠 Análise DatAI", "🕸️ Grafo"])

            with tab1:
                nomes = {
                    "shoulderLangle": "Ombro Esquerdo",
                    "shoulderRangle": "Ombro Direito",
                    "elbowLangle": "Cotovelo Esquerdo",
                    "elbowRangle": "Cotovelo Direito",
                    "kneeLangle": "Joelho Esquerdo",
                    "kneeRangle": "Joelho Direito"
                }

                colunas_disponiveis = [col for col in nomes if col in data.columns]

                escolha_articulacoes = st.multiselect(
                    "Selecione as articulações desejadas:",
                    options=[nomes[c] for c in colunas_disponiveis],
                    max_selections=6
                )

                if escolha_articulacoes:
                    col_map = {v: k for k, v in nomes.items()}
                    colunas_escolhidas = [col_map[n] for n in escolha_articulacoes]

                    st.markdown("# :material/keep: Estatísticas")
                    with st.container(border=False):
                        for coluna in colunas_escolhidas:
                            st.markdown(f"## **{nomes[coluna]}**")
                            cols = st.columns([0.4, 0.02, 0.4])
                            with cols[0].container(border=False):
                                st.metric("Valor Máximo", f"{data[coluna].max():.1f}°")
                                st.metric("Valor Mínimo", f"{data[coluna].min():.1f}°")
                            with cols[1].container(border=False):
                                st.html(    
                                    '''
                                        <div class="divider-vertical-line"></div>
                                        <style>
                                            .divider-vertical-line {
                                                border-left: 2px solid rgba(49, 51, 63, 0.2);
                                                height: 180px;  
                                                margin: auto;
                                            }
                                        </style>
                                    '''
                                )
                            with cols[2].container(border=False):
                                st.metric("Média", f"{data[coluna].mean():.1f}°")
                                st.metric("Amplitude", f"{(data[coluna].max() - data[coluna].min()):.1f}°")
                            st.divider()

                    st.line_chart(data[colunas_escolhidas], width=1390)
                    st.divider()

                    st.markdown("# :material/radar: Radar de Intensidade de Movimento")
                    all_art = [col for col in nomes if col in data.columns]
                    categories = []
                    values = []

                    for joint_key in all_art:
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

                # with st.expander("Tempos com ângulos extremos"):
                #     nomes = {
                #         "shoulderLangle": "Ombro Esquerdo",
                #         "shoulderRangle": "Ombro Direito",
                #         "elbowLangle": "Cotovelo Esquerdo",
                #         "elbowRangle": "Cotovelo Direito",
                #         "kneeLangle": "Joelho Esquerdo",
                #         "kneeRangle": "Joelho Direito"
                #     }

                #     escolha_nome = st.selectbox("Escolha a articulação para exibir:", list(nomes.values()))

                #     joint = None
                #     for key, nome in nomes.items():
                #         if nome == escolha_nome:
                #             joint = key
                #             break

                #     q3 = data[joint].quantile(2 / 3)
                #     max_val = data[joint].max()

                #     df_sel = data.loc[(data[joint] >= q3) & (data[joint] <= max_val), ['time', joint]].copy()

                #     df_sel['time_sec'] = ((df_sel['time'] - data['time'].min()) / 40).round(2)
                #     df_sel['angle'] = df_sel[joint].round(1)

                #     df_sel = df_sel.sort_values(by='angle', ascending=False).drop_duplicates(subset='time_sec')
                #     df_sel.sort_values(by='time_sec', inplace=True)

                #     if not df_sel.empty:
                #         st.markdown(f"{len(df_sel)} ocorrência(s) com ângulo elevado!")
                #         for _, row in df_sel.iterrows():
                #             st.markdown(f"- Em {row['time_sec']}s: ângulo de {row['angle']}°")
                #     else:
                #         st.markdown("- Nenhum valor no 3º intervalo.")

    
    inner.__name__ = f"paciente_{nome_completo.replace(' ', '_').lower()}"
    return inner

def home():
    st.set_page_config(page_title="Home", page_icon=":material/home:", layout="wide")
    st.markdown("## :material/waving_hand: Bem-vindo ao DatAI App!")
    st.write(f"Olá! Estamos felizes em tê-lo(a) aqui.")
    st.write("""
        O Datai App é sua ferramenta para uma análise aprofundada de movimentos. 
        Aqui você pode:
        - Visualizar estatísticas detalhadas de diversas articulações.
        - Analisar a intensidade de movimento através de gráficos de radar.
        - Explorar o movimento em 2D usando um grafo interativo.
        - Cadastrar novos pacientes e gerenciar seus dados.
    """)
    st.info("💡 Para começar, **selecione um paciente** no menu lateral ou **adicione um novo paciente** se necessário.")
    #st.image("https://via.placeholder.com/600x200?text=Sua+Imagem+de+Boas-Vindas+aqui", caption="Análise de movimento inteligente.")

def login():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""

    ph_login = st.container()

    if not st.session_state.get("authenticated", False):
        # não está logado
        ph_login.markdown("# :material/waving_hand: Bem-vindo!")
        supabase_connection = login_form()
        return
    else:
        # está logado
        if not st.session_state.get("toast_shown", False):
            if st.session_state.get("username"):
                msg = st.toast("Carregando...")
                time.sleep(1)
                msg.toast('Preparando...')
                time.sleep(1)
                msg.toast(f"Bem-vindo {st.session_state['username']}!", icon=":material/check:")
            else:
                msg = st.toast("Carregando...")
                time.sleep(1)
                msg.toast('Preparando...')
                time.sleep(1)
                msg.toast(f"Bem-vindo!", icon=":material/check:")
            st.session_state["toast_shown"] = True
        
        ph_login.empty()
        res = supabase.table("pacientes").select("nome, sobrenome").execute()
        pacientes_data = res.data or []
        logo_big = "logo_big.png"
        logo_small = "logo_small.png"
        st.logo(logo_big, icon_image=logo_small)

        pages = {
            "Minha Conta": [
                st.Page(home, title="Home", icon=":material/home:"),
            ],
            "Pacientes": [
                *[
                    st.Page(
                        patient(f"{p['nome']} {p['sobrenome']}"),
                        title=f"{p['nome']} {p['sobrenome']}",
                        icon=":material/person:"
                    )
                    for p in pacientes_data
                ],
                st.Page(create_pacientes, title="Adicionar Pacientes", icon=":material/manage_accounts:"),
            ]
        }

        nav = st.navigation(pages, position="sidebar", expanded=True)
        nav.run()
        return

login()
