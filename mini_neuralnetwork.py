import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import io
import random
import time
from supabase import create_client
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Função para slugificar nomes (necessária para o código)
def slugify(text):
    """Converte texto para slug (sem acentos, espaços, etc.)"""
    import re
    import unicodedata
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text.lower()

# Configuração do Supabase
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except:
    st.error("⚠️ Configuração do Supabase não encontrada. Verifique os secrets.")
    supabase = None

# Definição dos intents (mantendo os dados originais)
intents = {
    # JOELHO
    "maior_angulo_joelho_esquerdo": [
        "qual o maior ângulo do joelho esquerdo",
        "ângulo máximo do joelho esquerdo",
        "maior valor do joelho L",
        "pico do joelho esquerdo",
        "joelho esquerdo valor mais alto",
        "joelho L ângulo maior",
        "maior medição do joelho esquerdo",
        "ângulo mais aberto do joelho esquerdo",
        "extensão máxima do joelho esquerdo",
        "valor máximo joelho esquerdo",
        "joelho L maior ângulo registrado",
        "ângulo maior joelho L",
        "ângulo top do joelho esquerdo",
        "maior grau do joelho esquerdo",
        "qual foi o pico do joelho esquerdo",
        "joelho L com maior abertura",
        "joelho esquerdo ângulo mais alto",
        "joelho L ângulo top",
        "joelho esquerdo ângulo extremo",
        "valor mais alto atingido pelo joelho esquerdo"
    ],
    "menor_angulo_joelho_esquerdo": [
        "menor ângulo do joelho esquerdo",
        "ângulo mínimo do joelho esquerdo",
        "piso do joelho esquerdo",
        "valor mais baixo do joelho L",
        "joelho esquerdo menor valor",
        "ângulo mais fechado do joelho esquerdo",
        "menor grau joelho L",
        "qual foi o valor mínimo do joelho esquerdo",
        "joelho esquerdo menor ângulo registrado",
        "joelho L ângulo mínimo",
        "ângulo mais curto do joelho esquerdo",
        "valor mais baixo joelho esquerdo",
        "joelho L grau mais baixo",
        "qual o menor grau do joelho esquerdo",
        "joelho esquerdo valor inferior",
        "joelho esquerdo com menor abertura",
        "joelho esquerdo ângulo mais estreito",
        "joelho esquerdo ângulo mais reduzido",
        "ponto mais baixo do joelho esquerdo",
        "joelho esquerdo ângulo mínimo registrado"
    ],
    "media_angulo_joelho_esquerdo": [
        "qual a média do joelho esquerdo",
        "média ângulo joelho L",
        "valor médio do joelho esquerdo",
        "joelho L média",
        "média de ângulos do joelho esquerdo",
        "qual o valor médio do joelho esquerdo",
        "joelho esquerdo média angular",
        "joelho esquerdo ângulo médio",
        "média de movimentação do joelho esquerdo",
        "valor médio joelho L",
        "joelho esquerdo ângulo central",
        "média geral joelho esquerdo",
        "ângulo médio registrado joelho esquerdo",
        "grau médio do joelho esquerdo",
        "joelho esquerdo ângulo padrão",
        "qual a média de movimento do joelho esquerdo",
        "qual é a média de graus do joelho esquerdo",
        "média dos valores do joelho esquerdo",
        "média aritmética do joelho esquerdo",
        "joelho esquerdo valor de média"
    ],
    "amplitude_movimento_joelho_esquerdo": [
        "qual a amplitude do joelho esquerdo",
        "amplitude total do joelho esquerdo",
        "quanto o joelho esquerdo se movimentou",
        "movimento total do joelho esquerdo",
        "diferença entre maior e menor ângulo do joelho esquerdo",
        "alcance do joelho esquerdo",
        "joelho esquerdo amplitude de movimento",
        "extensão do movimento do joelho esquerdo",
        "quanto variou o ângulo do joelho esquerdo",
        "variação angular do joelho esquerdo",
        "amplitude de rotação do joelho esquerdo",
        "movimento articular do joelho esquerdo",
        "graus percorridos pelo joelho esquerdo",
        "limite de movimento do joelho esquerdo",
        "abertura total do joelho esquerdo",
        "quantos graus o joelho esquerdo se moveu",
        "grau de flexão e extensão do joelho esquerdo",
        "joelho esquerdo variação de ângulo",
        "quanta movimentação teve o joelho esquerdo",
        "diferença de graus do joelho esquerdo"
    ],

    "maior_angulo_joelho_direito": [
        "qual o maior ângulo do joelho direito",
        "ângulo máximo do joelho direito",
        "maior valor do joelho R",
        "pico do joelho direito",
        "joelho direito valor mais alto",
        "joelho R ângulo maior",
        "maior medição do joelho direito",
        "ângulo mais aberto do joelho direito",
        "extensão máxima do joelho direito",
        "valor máximo joelho direito",
        "joelho R maior ângulo registrado",
        "ângulo maior joelho R",
        "ângulo top do joelho direito",
        "maior grau do joelho direito",
        "qual foi o pico do joelho direito",
        "joelho R com maior abertura",
        "joelho direito ângulo mais alto",
        "joelho R ângulo top",
        "joelho direito ângulo extremo",
        "valor mais alto atingido pelo joelho direito"
    ],
    "menor_angulo_joelho_direito": [
        "menor ângulo do joelho direito",
        "ângulo mínimo do joelho direito",
        "piso do joelho direito",
        "valor mais baixo do joelho R",
        "joelho direito menor valor",
        "ângulo mais fechado do joelho direito",
        "menor grau joelho R",
        "qual foi o valor mínimo do joelho direito",
        "joelho direito menor ângulo registrado",
        "joelho R ângulo mínimo",
        "ângulo mais curto do joelho direito",
        "valor mais baixo joelho direito",
        "joelho R grau mais baixo",
        "qual o menor grau do joelho direito",
        "joelho direito valor inferior",
        "joelho direito com menor abertura",
        "joelho direito ângulo mais estreito",
        "joelho direito ângulo mais reduzido",
        "ponto mais baixo do joelho direito",
        "joelho direito ângulo mínimo registrado"
    ],
    "media_angulo_joelho_direito": [
        "qual a média do joelho direito",
        "média ângulo joelho R",
        "valor médio do joelho direito",
        "joelho R média",
        "média de ângulos do joelho direito",
        "qual o valor médio do joelho direito",
        "joelho direito média angular",
        "joelho direito ângulo médio",
        "média de movimentação do joelho direito",
        "valor médio joelho R",
        "joelho direito ângulo central",
        "média geral joelho direito",
        "ângulo médio registrado joelho direito",
        "grau médio do joelho direito",
        "joelho direito ângulo padrão",
        "qual a média de movimento do joelho direito",
        "qual é a média de graus do joelho direito",
        "média dos valores do joelho direito",
        "média aritmética do joelho direito",
        "joelho direito valor de média"
    ],
    "amplitude_movimento_joelho_direito": [
        "qual a amplitude do joelho direito",
        "amplitude total do joelho direito",
        "quanto o joelho direito se movimentou",
        "movimento total do joelho direito",
        "diferença entre maior e menor ângulo do joelho direito",
        "alcance do joelho direito",
        "joelho direito amplitude de movimento",
        "extensão do movimento do joelho direito",
        "quanto variou o ângulo do joelho direito",
        "variação angular do joelho direito",
        "amplitude de rotação do joelho direito",
        "movimento articular do joelho direito",
        "graus percorridos pelo joelho direito",
        "limite de movimento do joelho direito",
        "abertura total do joelho direito",
        "quantos graus o joelho direito se moveu",
        "grau de flexão e extensão do joelho direito",
        "joelho direito variação de ângulo",
        "quanta movimentação teve o joelho direito",
        "diferença de graus do joelho direito"
    ],

    # OMBRO ESQUERDO
    "maior_angulo_ombro_esquerdo": [
        "qual o maior ângulo do ombro esquerdo",
        "ângulo máximo do ombro esquerdo",
        "maior valor do ombro L",
        "pico do ombro esquerdo",
        "ombro esquerdo valor mais alto",
        "ombro L ângulo maior"
    ],
    "menor_angulo_ombro_esquerdo": [
        "qual o menor ângulo do ombro esquerdo",
        "ângulo mínimo do ombro esquerdo",
        "menor valor do ombro L",
        "piso do ombro esquerdo",
        "ombro esquerdo menor valor"
    ],
    "media_angulo_ombro_esquerdo": [
        "qual a média do ombro esquerdo",
        "média ângulo ombro L",
        "valor médio do ombro esquerdo",
        "ombro L média"
    ],
    "amplitude_movimento_ombro_esquerdo": [
        "qual a amplitude do ombro esquerdo",
        "amplitude total do ombro esquerdo",
        "quanto o ombro esquerdo se movimentou"
    ],

    # OMBRO DIREITO
    "maior_angulo_ombro_direito": [
        "qual o maior ângulo do ombro direito",
        "ângulo máximo do ombro direito",
        "maior valor do ombro R",
        "pico do ombro direito"
    ],
    "menor_angulo_ombro_direito": [
        "menor ângulo do ombro direito",
        "ângulo mínimo do ombro direito",
        "piso do ombro direito"
    ],
    "media_angulo_ombro_direito": [
        "qual a média do ombro direito",
        "média ângulo ombro R",
        "valor médio do ombro direito"
    ],
    "amplitude_movimento_ombro_direito": [
        "qual a amplitude do ombro direito",
        "amplitude total do ombro direito"
    ],

    # COTOVELO ESQUERDO
    "maior_angulo_cotovelo_esquerdo": [
        "qual o maior ângulo do cotovelo esquerdo",
        "ângulo máximo do cotovelo esquerdo",
        "maior valor do cotovelo L"
    ],
    "menor_angulo_cotovelo_esquerdo": [
        "menor ângulo do cotovelo esquerdo",
        "ângulo mínimo do cotovelo esquerdo"
    ],
    "media_angulo_cotovelo_esquerdo": [
        "qual a média do cotovelo esquerdo",
        "média ângulo cotovelo L"
    ],
    "amplitude_movimento_cotovelo_esquerdo": [
        "qual a amplitude do cotovelo esquerdo",
        "amplitude total do cotovelo L"
    ],

    # COTOVELO DIREITO
    "maior_angulo_cotovelo_direito": [
        "qual o maior ângulo do cotovelo direito",
        "ângulo máximo do cotovelo direito"
    ],
    "menor_angulo_cotovelo_direito": [
        "menor ângulo do cotovelo direito",
        "ângulo mínimo do cotovelo direito"
    ],
    "media_angulo_cotovelo_direito": [
        "qual a média do cotovelo direito",
        "média ângulo cotovelo R"
    ],
    "amplitude_movimento_cotovelo_direito": [
        "qual a amplitude do cotovelo direito",
        "amplitude total do cotovelo R"
    ],

    # SAUDAÇÃO E ENCERRAMENTO
    "saudacao": [
        "oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí", "fala", "salve", "opa",
        "como vai", "tudo bem?", "beleza?", "alô", "bom te ver", "bom dia pra você",
        "olá, tudo certo?", "oi, tudo bem?", "bom encontrar você", "e aí, beleza?"
    ],
    "encerramento": [
        "tchau", "obrigado", "valeu", "até mais", "até logo", "fui", "flw",
        "até a próxima", "fica com Deus", "até breve", "agradecido", "até já",
        "até amanhã", "foi bom falar com você", "até a gente se falar de novo",
        "te vejo depois", "se cuida", "boa noite e bons sonhos", "até qualquer hora", "até já já"
    ]
}

# Classe para vectorização BERT
class BERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.eval()

    def transform(self, X):
        embeddings = []
        for sentence in X:
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
                embeddings.append(cls_embedding)
        return embeddings

    def fit(self, X, y=None):
        return self

# Inicialização do modelo (com cache)
@st.cache_resource
def load_models():
    """Carrega os modelos BERT e treina o pipeline"""
    try:
        # Preparar dados de treinamento
        texts, labels = [], []
        for label, exemplos in intents.items():
            texts.extend(exemplos)
            labels.extend([label] * len(exemplos))

        # Carregar modelos BERT
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")

        # Preparar pipeline
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)

        pipeline_model = Pipeline([
            ("bert_vec", BERTVectorizer(tokenizer, model)),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        
        # Treinar pipeline
        pipeline_model.fit(texts, y_encoded)
        
        return pipeline_model, label_encoder
        
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {str(e)}")
        return None, None

# Função principal do chatbot
def chatbot_brain(prompt, file_patient, username, name_patient):
    """Processa a pergunta do usuário e retorna resposta baseada nos dados"""
    
    # Carregar modelos
    pipeline_model, label_encoder = load_models()
    if pipeline_model is None:
        return "❌ Erro ao carregar o modelo de IA. Tente novamente."
    
    df = None
    
    # Processar arquivo do paciente
    try:
        if isinstance(file_patient, str) and supabase:
            # Baixar do Supabase
            doc = supabase.storage.from_("pacientes").download(file_patient)
            text_data = io.StringIO(doc.decode("utf-8"))
            df = pd.read_csv(text_data)
        elif isinstance(file_patient, pd.DataFrame):
            df = file_patient
    except Exception as e:
        return f"❌ Erro ao processar arquivo: {str(e)}"

    # Classificar intenção
    try:
        vec = [prompt]
        pred = pipeline_model.predict(vec)
        intent = label_encoder.inverse_transform(pred)[0]
    except Exception as e:
        return f"❌ Erro ao processar pergunta: {str(e)}"
    
    username = username.capitalize() if username else "Usuário"
    name_patient = name_patient if name_patient else "paciente"
    
    # Verificar se há dados
    if df is None or df.empty:
        if intent in ["saudacao", "encerramento"]:
            if intent == "saudacao":
                return f"👋 Olá {username}! Como posso ajudar com os dados de {name_patient}?"
            else:
                return f"👋 Até logo {username}, estarei aqui se precisar novamente!"
        else:
            return f"📁 {username}, você precisa fazer upload do arquivo CSV/XLSX primeiro!"

    # Verificar colunas necessárias
    required_columns = ['kneeLangle', 'kneeRangle', 'shoulderLangle', 'shoulderRangle', 'elbowLangle', 'elbowRangle']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return f"❌ Colunas faltando no arquivo: {', '.join(missing_columns)}"

    # Processar intenções
    try:
        # JOELHO ESQUERDO
        if intent == "maior_angulo_joelho_esquerdo":
            valor = df['kneeLangle'].max()
            return f"📊 O maior ângulo do joelho esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "menor_angulo_joelho_esquerdo":
            valor = df['kneeLangle'].min()
            return f"📊 O menor ângulo do joelho esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "media_angulo_joelho_esquerdo":
            valor = df['kneeLangle'].mean()
            return f"📊 A média dos ângulos do joelho esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "amplitude_movimento_joelho_esquerdo":
            amplitude = df["kneeLangle"].max() - df["kneeLangle"].min()
            return f"📊 A amplitude de movimento do joelho esquerdo de {name_patient} é **{amplitude:.2f}°**"

        # JOELHO DIREITO
        elif intent == "maior_angulo_joelho_direito":
            valor = df['kneeRangle'].max()
            return f"📊 O maior ângulo do joelho direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "menor_angulo_joelho_direito":
            valor = df['kneeRangle'].min()
            return f"📊 O menor ângulo do joelho direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "media_angulo_joelho_direito":
            valor = df['kneeRangle'].mean()
            return f"📊 A média dos ângulos do joelho direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "amplitude_movimento_joelho_direito":
            amplitude = df["kneeRangle"].max() - df["kneeRangle"].min()
            return f"📊 A amplitude de movimento do joelho direito de {name_patient} é **{amplitude:.2f}°**"
        
        # OMBRO ESQUERDO
        elif intent == "maior_angulo_ombro_esquerdo":
            valor = df['shoulderLangle'].max()
            return f"📊 O maior ângulo do ombro esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "menor_angulo_ombro_esquerdo":
            valor = df['shoulderLangle'].min()
            return f"📊 O menor ângulo do ombro esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "media_angulo_ombro_esquerdo":
            valor = df['shoulderLangle'].mean()
            return f"📊 A média dos ângulos do ombro esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "amplitude_movimento_ombro_esquerdo":
            amplitude = df["shoulderLangle"].max() - df["shoulderLangle"].min()
            return f"📊 A amplitude de movimento do ombro esquerdo de {name_patient} é **{amplitude:.2f}°**"
        
        # OMBRO DIREITO
        elif intent == "maior_angulo_ombro_direito":
            valor = df['shoulderRangle'].max()
            return f"📊 O maior ângulo do ombro direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "menor_angulo_ombro_direito":
            valor = df['shoulderRangle'].min()
            return f"📊 O menor ângulo do ombro direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "media_angulo_ombro_direito":
            valor = df['shoulderRangle'].mean()
            return f"📊 A média dos ângulos do ombro direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "amplitude_movimento_ombro_direito":
            amplitude = df["shoulderRangle"].max() - df["shoulderRangle"].min()
            return f"📊 A amplitude de movimento do ombro direito de {name_patient} é **{amplitude:.2f}°**"
        
        # COTOVELO ESQUERDO
        elif intent == "maior_angulo_cotovelo_esquerdo":
            valor = df['elbowLangle'].max()
            return f"📊 O maior ângulo do cotovelo esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "menor_angulo_cotovelo_esquerdo":
            valor = df['elbowLangle'].min()
            return f"📊 O menor ângulo do cotovelo esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "media_angulo_cotovelo_esquerdo":
            valor = df['elbowLangle'].mean()
            return f"📊 A média dos ângulos do cotovelo esquerdo de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "amplitude_movimento_cotovelo_esquerdo":
            amplitude = df["elbowLangle"].max() - df["elbowLangle"].min()
            return f"📊 A amplitude de movimento do cotovelo esquerdo de {name_patient} é **{amplitude:.2f}°**"
        
        # COTOVELO DIREITO
        elif intent == "maior_angulo_cotovelo_direito":
            valor = df['elbowRangle'].max()
            return f"📊 O maior ângulo do cotovelo direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "menor_angulo_cotovelo_direito":
            valor = df['elbowRangle'].min()
            return f"📊 O menor ângulo do cotovelo direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "media_angulo_cotovelo_direito":
            valor = df['elbowRangle'].mean()
            return f"📊 A média dos ângulos do cotovelo direito de {name_patient} é **{valor:.2f}°**"
            
        elif intent == "amplitude_movimento_cotovelo_direito":
            amplitude = df["elbowRangle"].max() - df["elbowRangle"].min()
            return f"📊 A amplitude de movimento do cotovelo direito de {name_patient} é **{amplitude:.2f}°**"

        # SAUDAÇÃO E ENCERRAMENTO
        elif intent == "saudacao":
            return f"👋 Olá {username}! Como posso ajudar com os dados de {name_patient}?"
            
        elif intent == "encerramento":
            return f"👋 Até logo {username}, estarei aqui se precisar novamente!"

        # CASO NÃO RECONHECIDO
        else:
            return f"🤔 Desculpe {username}, não entendi sua pergunta sobre {name_patient}. Tente perguntar sobre ângulos dos joelhos, ombros ou cotovelos."
            
    except Exception as e:
        return f"❌ Erro ao processar dados: {str(e)}"

