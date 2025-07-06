import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch, io, random
from supabase import create_client

# Supabase config
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

intents = {
    # JOELHO
    "maior_angulo_joelho_esquerdo": [
        "qual o maior ângulo do joelho esquerdo",
        "ângulo máximo do joelho esquerdo",
        "maior valor do joelho L",
        "pico do joelho esquerdo"
    ],
    "menor_angulo_joelho_esquerdo": [
        "menor ângulo do joelho esquerdo",
        "ângulo mínimo do joelho esquerdo",
        "piso do joelho esquerdo",
        "valor mais baixo do joelho L"
    ],
    "media_angulo_joelho_esquerdo": [
        "qual a média do joelho esquerdo",
        "média ângulo joelho L",
        "valor médio do joelho esquerdo"
    ],
    "maior_angulo_joelho_direito": [
        "qual o maior ângulo do joelho direito",
        "ângulo máximo do joelho direito",
        "maior valor do joelho R"
    ],
    "menor_angulo_joelho_direito": [
        "menor ângulo do joelho direito",
        "ângulo mínimo do joelho direito",
        "piso do joelho direito",
        "valor mais baixo do joelho R"
    ],
    "media_angulo_joelho_direito": [
        "qual a média do joelho direito",
        "média ângulo joelho R",
        "valor médio do joelho direito"
    ],

    # OMBRO
    "maior_angulo_ombro_esquerdo": [
        "maior ângulo do ombro esquerdo",
        "pico do ombro esquerdo",
        "ângulo máximo do ombro L"
    ],
    "menor_angulo_ombro_esquerdo": [
        "menor ângulo do ombro esquerdo",
        "piso do ombro esquerdo",
        "ângulo mínimo do ombro L"
    ],
    "media_angulo_ombro_esquerdo": [
        "qual a média do ombro esquerdo",
        "média ângulo ombro L",
        "valor médio do ombro esquerdo"
    ],
    "maior_angulo_ombro_direito": [
        "maior ângulo do ombro direito",
        "pico do ombro direito",
        "ângulo máximo do ombro R"
    ],
    "menor_angulo_ombro_direito": [
        "menor ângulo do ombro direito",
        "piso do ombro direito",
        "ângulo mínimo do ombro R"
    ],
    "media_angulo_ombro_direito": [
        "qual a média do ombro direito",
        "média ângulo ombro R",
        "valor médio do ombro direito"
    ],

    # COTOVELO
    "maior_angulo_cotovelo_esquerdo": [
        "maior ângulo do cotovelo esquerdo",
        "pico do cotovelo esquerdo",
        "ângulo máximo do cotovelo L"
    ],
    "menor_angulo_cotovelo_esquerdo": [
        "menor ângulo do cotovelo esquerdo",
        "piso do cotovelo esquerdo",
        "ângulo mínimo do cotovelo L"
    ],
    "media_angulo_cotovelo_esquerdo": [
        "qual a média do cotovelo esquerdo",
        "média ângulo cotovelo L",
        "valor médio do cotovelo esquerdo"
    ],
    "maior_angulo_cotovelo_direito": [
        "maior ângulo do cotovelo direito",
        "pico do cotovelo direito",
        "ângulo máximo do cotovelo R"
    ],
    "menor_angulo_cotovelo_direito": [
        "menor ângulo do cotovelo direito",
        "piso do cotovelo direito",
        "ângulo mínimo do cotovelo R"
    ],
    "media_angulo_cotovelo_direito": [
        "qual a média do cotovelo direito",
        "média ângulo cotovelo R",
        "valor médio do cotovelo direito"
    ],

    # SAUDAÇÃO E ENCERRAMENTO
    "saudacao": [
        "oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí", "fala", "salve", "opa"
    ],
    "encerramento": [
        "tchau", "obrigado", "valeu", "até mais", "até logo", "fui", "flw"
    ]
}

# Treinamento
texts, labels = [], []
for label, exemplos in intents.items():
    texts.extend(exemplos)
    labels.extend([label] * len(exemplos))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

pipeline_model = Pipeline([
    ("bert_vec", BERTVectorizer(tokenizer, model)),
    ("clf", LogisticRegression(max_iter=1000))
])
pipeline_model.fit(texts, y_encoded)

# Função principal
def chatbot_brain(prompt, file_patient, username, name_patient):
    df = None
    if isinstance(file_patient, str):
        doc = supabase.storage.from_("pacientes").download(file_patient)
        text_data = io.StringIO(doc.decode("utf-8"))
        df = pd.read_csv(text_data)
    elif isinstance(file_patient, pd.DataFrame):
        df = file_patient

    vec = [prompt]
    pred = pipeline_model.predict(vec)
    intent = label_encoder.inverse_transform(pred)[0]

    if df is not None:
        # JOELHO
        if intent == "maior_angulo_joelho_esquerdo":
            return f"O maior ângulo do joelho esquerdo é {df['kneeLangle'].max():.2f}º."
        elif intent == "menor_angulo_joelho_esquerdo":
            return f"O menor ângulo do joelho esquerdo é {df['kneeLangle'].min():.2f}º."
        elif intent == "media_angulo_joelho_esquerdo":
            return f"A média do ângulo do joelho esquerdo é {df['kneeLangle'].mean():.2f}º."
        elif intent == "maior_angulo_joelho_direito":
            return f"O maior ângulo do joelho direito é {df['kneeRangle'].max():.2f}º."
        elif intent == "menor_angulo_joelho_direito":
            return f"O menor ângulo do joelho direito é {df['kneeRangle'].min():.2f}º."
        elif intent == "media_angulo_joelho_direito":
            return f"A média do ângulo do joelho direito é {df['kneeRangle'].mean():.2f}º."

        # OMBRO
        elif intent == "maior_angulo_ombro_esquerdo":
            return f"O maior ângulo do ombro esquerdo é {df['shoulderLangle'].max():.2f}º."
        elif intent == "menor_angulo_ombro_esquerdo":
            return f"O menor ângulo do ombro esquerdo é {df['shoulderLangle'].min():.2f}º."
        elif intent == "media_angulo_ombro_esquerdo":
            return f"A média do ângulo do ombro esquerdo é {df['shoulderLangle'].mean():.2f}º."
        elif intent == "maior_angulo_ombro_direito":
            return f"O maior ângulo do ombro direito é {df['shoulderRangle'].max():.2f}º."
        elif intent == "menor_angulo_ombro_direito":
            return f"O menor ângulo do ombro direito é {df['shoulderRangle'].min():.2f}º."
        elif intent == "media_angulo_ombro_direito":
            return f"A média do ângulo do ombro direito é {df['shoulderRangle'].mean():.2f}º."

        # COTOVELO
        elif intent == "maior_angulo_cotovelo_esquerdo":
            return f"O maior ângulo do cotovelo esquerdo é {df['elbowLangle'].max():.2f}º."
        elif intent == "menor_angulo_cotovelo_esquerdo":
            return f"O menor ângulo do cotovelo esquerdo é {df['elbowLangle'].min():.2f}º."
        elif intent == "media_angulo_cotovelo_esquerdo":
            return f"A média do ângulo do cotovelo esquerdo é {df['elbowLangle'].mean():.2f}º."
        elif intent == "maior_angulo_cotovelo_direito":
            return f"O maior ângulo do cotovelo direito é {df['elbowRangle'].max():.2f}º."
        elif intent == "menor_angulo_cotovelo_direito":
            return f"O menor ângulo do cotovelo direito é {df['elbowRangle'].min():.2f}º."
        elif intent == "media_angulo_cotovelo_direito":
            return f"A média do ângulo do cotovelo direito é {df['elbowRangle'].mean():.2f}º."

        # SAUDAÇÃO E ENCERRAMENTO
        elif intent == "saudacao":
            return f"Olá {username}! Como posso ajudar com os dados de {name_patient}?"
        elif intent == "encerramento":
            return f"Até logo {username}, estarei aqui se precisar novamente!"

        # DEFAULT
        else:
            return f"Desculpe {username}, não entendi sua pergunta."

    else:
        return f"{username.capitalize()}, você precisa fazer upload do arquivo CSV/XLSX primeiro! ☝️"
