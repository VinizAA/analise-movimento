import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import time, random, unicodedata, re, io

from datetime import datetime, date
from typing import List
from mini_neuralnetwork import predict, extract, answer
from st_login_form import login_form, logout
from supabase import create_client
from dateutil.relativedelta import relativedelta

# 1. Config geral do app - só uma vez, no topo do arquivo
st.set_page_config(page_title="Datai App", page_icon=":material/home:", layout="centered")

# 2. Config Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

#configurações conta/autenticação
def init_auth_state():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "login_time" not in st.session_state:
        st.session_state["login_time"] = None

def restore_session_from_url():
    url_username = get_username_from_url()
    
    if url_username and not st.session_state["authenticated"]:
        try:
            user_check = supabase.table("user_sessions").select("username").eq("username", url_username).execute()
            
            if user_check.data:
                st.session_state["authenticated"] = True
                st.session_state["username"] = url_username
                st.session_state["login_time"] = datetime.now()
                return True
        except:
            st.session_state["authenticated"] = True
            st.session_state["username"] = url_username
            st.session_state["login_time"] = datetime.now()
            return True
    return False

def save_session_to_db(username):
    try:
        session_data = {
            "username": username,
            "login_time": datetime.now().isoformat(),
            "expires_at": (datetime.now() + relativedelta(days=7)).isoformat() 
        }
        
        supabase.table("user_sessions").upsert(session_data).execute()
    except Exception as e:

        print(f"Erro ao salvar sessão: {e}")

def clear_session_from_db(username):
    try:
        supabase.table("user_sessions").delete().eq("username", username).execute()
    except Exception as e:
        print(f"Erro ao limpar sessão: {e}")

def set_url_with_username(username):
    username_clean = slugify(username)
    st.query_params.update({"user": username_clean})

def get_username_from_url():
    return st.query_params.get("user", None)

def clear_url_params():
    if "user" in st.query_params:
        del st.query_params["user"]

#auxiliares
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

def slugify(value):
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^a-zA-Z0-9_-]', '_', value)
    return value

def show_analysis_tab(data: pd.DataFrame):
    """Função para reutilizar a aba de análise"""
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

        if len(colunas_escolhidas) == 1:
            st.markdown(f"# :material/show_chart: Variação do {nomes[colunas_escolhidas[0]]} ao longo do tempo")
        else:
            st.markdown(f"# :material/show_chart: Variação das articulações ao longo do tempo")

        fig = go.Figure()
        for coluna in colunas_escolhidas:
            fig.add_trace(go.Scatter(y=data[coluna], mode="lines", name=nomes[coluna]))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
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

def delete_patient(paciente_id, documento_url):
    try:
        if not paciente_id:
            st.error("ID do paciente é obrigatório")
            return False
        
        # Remover documento se existir
        if documento_url:
            try:
                supabase.storage.from_("pacientes").remove([documento_url])
            except Exception as storage_error:
                st.warning(f"Não foi possível remover o documento: {storage_error}")
        
        result = supabase.table("pacientes").delete().eq("id", paciente_id).execute()
        
        if hasattr(result, 'error') and result.error:
            st.error(f"Erro na exclusão: {result.error}")
            return False
        
        st.success("Paciente excluído com sucesso!", icon=":material/delete:")
        time.sleep(2)
        st.rerun()
        return True
            
    except Exception as e:  
        st.error(f"Erro ao excluir paciente: {e}")
        return False
    
#páginas
def create_pacientes():
    st.set_page_config(page_title="Cadastro de Paciente", page_icon=":material/add:", layout="wide")
    st.markdown("# :material/add: Cadastro de pacientes")

    with st.form("form_paciente"):
        nome = st.text_input("Nome")    
        sobrenome = st.text_input("Sobrenome")
        sexo = st.selectbox("Sexos", ["Masculino", "Feminino", "Outro"])

        min_date = date.today() - relativedelta(years=120)
        date_nasc = st.date_input("Data de Nascimento (DD/MM/AAAA)", format="DD/MM/YYYY", max_value="today", min_value=min_date , value=None)

        documento = st.file_uploader("Documento dos movimentos", type=["csv", "xlsx"])

        st.container(height=5, border=False)
        if st.form_submit_button("Cadastrar"):
            if not all([nome, sobrenome, sexo, date_nasc, documento]):
                st.warning("Preencha todos os campos")
                return

            hoje = date.today()
            idade = hoje.year - date_nasc.year - ((hoje.month, hoje.day) < (date_nasc.month, date_nasc.day))

            with st.spinner("Enviando dados..."):
                doc_bytes = documento.read()
                nome_limpo = slugify(nome)
                sobrenome_limpo = slugify(sobrenome)
                doc_path = f"pacientes/documentos/{nome_limpo}_{sobrenome_limpo}_{int(time.time())}.{documento.name.split('.')[-1]}"
                supabase.storage.from_("pacientes").upload(doc_path, doc_bytes, {"content-type": documento.type})

                data = {
                    "nome": nome,
                    "sobrenome": sobrenome,
                    "sexo": sexo,
                    "data_nascimento": date_nasc.isoformat(),
                    "idade": idade,
                    "documento_url": doc_path
                }
            
            time.sleep(2)
            doc = supabase.table("pacientes").insert(data).execute()
            if doc.data:
                st.success("Paciente cadastrado com sucesso!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("Erro ao cadastrar paciente.")

def patient(nome_completo, uploaded_file, paciente_id):
    def inner():
        st.set_page_config(page_title=f"{nome_completo}", page_icon=":material/person:", layout="wide")
        st.title(f"Análise de movimento de **{nome_completo}**")
        
        try:
            doc = supabase.storage.from_("pacientes").download(uploaded_file)
            data = pd.read_csv(io.BytesIO(doc))

            key_loading = f"loading_shown_{slugify(nome_completo)}"
            if not st.session_state.get(key_loading, False):
                my_bar = st.progress(0, text="Trazendo documento do paciente...")
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text="Trazendo documento do paciente...")
                time.sleep(1)
                my_bar.empty()

                ph_check = st.empty()
                ph_check.success(":material/check: Dados do paciente carregados com sucesso!")

            csv_file = data.to_csv().encode("utf-8")

            cols_main = st.columns([3, 1, 2])
            with cols_main[0]:
                cols_buttons = st.columns([0.6, 0.4], gap=None, vertical_alignment="bottom")
                with cols_buttons[0]:
                    st.download_button(
                        label=f"Baixar documento de {nome_completo}",
                        data=csv_file,
                        file_name=f"{slugify(nome_completo)}.csv",
                        mime="text/csv",
                        icon=":material/download:",
                    )
        except Exception as e:
            st.error(f":material/close: Erro ao carregar dados do paciente: {e}")
            return
        
        with cols_buttons[1]:
            with st.popover("Excluir Paciente", icon=":material/delete:"):
                st.markdown("### ⚠️ Atenção!") 
                st.write("Você está prestes a **:red[EXCLUIR PERMANENTEMENTE]** este paciente.")
                st.write("Esta ação é **:red[IRREVERSÍVEL]** e todos os dados associados serão perdidos.")
                                        
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Sim, excluir definitivamente", type="primary", use_container_width=True):
                        delete_patient(paciente_id, uploaded_file) 
                        st.success("Paciente excluído com sucesso!") 
                        st.rerun()
                with col2:
                    if st.button("Manter paciente", use_container_width=True):
                        st.info("Exclusão cancelada. O paciente permanece na sua lista.")
                        st.rerun()

        if data is not None: 
            tab1, tab2 = st.tabs(["🧠 Análise", "🕸️ Grafo"])

            with tab1:
                show_analysis_tab(data)

            with tab2:
                chosen_time = st.slider("Escolha o tempo (s)",
                                        min_value=0.0,
                                        max_value=((data['time'].max()) - (data['time'].min())) / 40,
                                        value=0.0,
                                        step=0.1,
                                        format="%.2f")
                plot_graph(data, chosen_time)

        if not st.session_state.get(key_loading, False):
            time.sleep(1)
            ph_check.empty()

        st.session_state[key_loading] = True
    
    inner.__name__ = f"paciente_{nome_completo.replace(' ', '_').lower()}"
    return inner

def home(nome_completo):
    st.set_page_config(page_title="Home", page_icon=":material/home:", layout="wide")
    st.markdown("## :material/waving_hand: Bem-vindo ao DatAI App!")
    st.write(f"Olá {nome_completo}! Estamos felizes em tê-lo(a) aqui.")
    st.write("""
        O Datai App é sua ferramenta para uma análise aprofundada de movimentos. 
        Aqui você pode:
        - Visualizar estatísticas detalhadas de diversas articulações.
        - Analisar a intensidade de movimento através de gráficos de radar.
        - Explorar o movimento em 2D usando um grafo interativo.
        - Cadastrar novos pacientes e gerenciar seus dados.
    """)

def main_app_guest():
    logo_big = "logo_big.png"
    logo_small = "logo_small.png"
    st.logo(logo_big, icon_image=logo_small)

    if not st.session_state.get("toast_shown", False):
        msg = st.toast("Carregando...")
        time.sleep(1)
        msg.toast("Preparando...")
        time.sleep(1)
        msg.toast(f"Bem-vindo!", icon=":material/check:")

        st.session_state["toast_shown"] = True

    st.set_page_config(page_title="Análise do movimento", page_icon=":material/analytics:", layout="wide")
    with st.sidebar:
        uploaded_file = st.file_uploader("Faça upload do arquivo", type=["csv", "xlsx"])
        st.info("Para começar, **adicione um arquivo** acima", icon=":material/arrow_upward:")
        data = None

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

    if data is not None: 
        st.set_page_config(page_title="Análise do movimento", page_icon=":material/analytics:", layout="wide")
        st.title(":material/analytics: Análise do movimento")
        tab1, tab2 = st.tabs(["🧠 Análise", "🕸️ Grafo"])

        with tab1:
            show_analysis_tab(data)

        with tab2:
            chosen_time = st.slider("Escolha o tempo (s)",
                                    min_value=0.0,
                                    max_value=((data['time'].max()) - (data['time'].min())) / 40,
                                    value=0.0,
                                    step=0.1,
                                    format="%.2f")
            plot_graph(data, chosen_time)
                
def main_app(nome_completo: str):
    logo_big = "logo_big.png"
    logo_small = "logo_small.png"
    st.logo(logo_big, icon_image=logo_small)

    if not st.session_state.get("toast_shown", False):
        user = st.session_state.get("username", "usuário")
        if user:
            msg = st.toast("Carregando...")
            time.sleep(1)
            msg.toast("Preparando...")
            time.sleep(1)
            msg.toast(f"Bem-vindo {user}!", icon=":material/check:")
        st.session_state["toast_shown"] = True

    pacientes_data = supabase.table("pacientes").select("id, nome, sobrenome, documento_url").execute().data or []

    def home_wrapper():
        home(nome_completo)

    pages = {
        "Minha Conta": [
            st.Page(home_wrapper, title="Home", icon=":material/home:")
        ],
        "Pacientes": []
    }

    for p in pacientes_data:
        if not p.get("documento_url"):
            continue
        
        nome_completo_p = f"{p['nome']} {p['sobrenome']}"
        uploaded_file = p['documento_url']

        pagina_func = patient(nome_completo_p, uploaded_file, p['id'])
        pagina_func.__name__ = f"paciente_{slugify(nome_completo_p)}_{p['id']}"
        
        pages["Pacientes"].append(
            st.Page(pagina_func, title=nome_completo_p, icon=":material/person:")
        )

    pages["Pacientes"].append(
        st.Page(create_pacientes, title="Adicionar Pacientes", icon=":material/manage_accounts:")
    )

    nav = st.navigation(pages, position="sidebar", expanded=True)
    nav.run()

def login():
    init_auth_state()
    
    session_restored = restore_session_from_url()
    
    if session_restored:
        username = st.session_state["username"]
        if username:
            main_app(username)
        else:
            main_app_guest()
        return
    
    if not st.session_state["authenticated"]:
        # Usuário ainda não logado
        st.markdown("# :material/waving_hand: Bem-vindo!")
        
        client_info = login_form()
        
        if client_info and client_info.get("username"):
            st.session_state["authenticated"] = True
            st.session_state["username"] = client_info["username"]
            st.session_state["login_time"] = datetime.now()
            
            set_url_with_username(client_info["username"])
            
            st.success(f"Login realizado com sucesso! Bem-vindo, {client_info['username']}!")
            time.sleep(1)
            st.rerun()
        else:
            return
    else:
        # Usuário já está logado
        username = st.session_state["username"]
        
        url_username = get_username_from_url()
        if url_username != slugify(username):
            set_url_with_username(username)

        if username:
            main_app(username)
        else:
            main_app_guest()

login()