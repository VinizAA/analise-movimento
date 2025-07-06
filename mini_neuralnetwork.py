import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import torch, io, random

from supabase import create_client

# Supabase config
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


intents = {
    "maior_angulo_joelho_esquerdo": [
        "qual o maior ângulo do joelho esquerdo",
        "ângulo máximo do joelho esquerdo",
        "maior valor do joelho L",
        "qual foi o pico do joelho esquerdo",
        "qual valor mais alto do joelho esquerdo",
        "joelho esquerdo chegou até quanto?",
        "qual foi o maior grau no joelho L",
        "me fala o ângulo máximo do joelho esquerdo",
        "qual o maior grau que o joelho esquerdo atingiu?",
        "qual a maior abertura do joelho esquerdo?",
        "pico de flexão do joelho esquerdo",
        "máximo do joelho esquerdo",
        "ângulo mais alto do joelho esquerdo",
        "recorde de ângulo do joelho esquerdo"
    ],

    "maior_angulo_joelho_direito": [
        "qual o maior ângulo do joelho direito",
        "ângulo máximo do joelho direito",
        "maior valor do joelho R",
        "joelho direito chegou até quanto?",
        "qual valor mais alto no joelho direito",
        "qual foi o pico do joelho direito",
        "quanto deu o ângulo máximo do joelho R",
        "me informa o maior ângulo do joelho direito",
        "qual o maior grau que o joelho direito fez?",
        "qual a maior abertura do joelho direito?",
        "pico de extensão do joelho direito",
        "máximo do joelho direito",
        "ângulo mais alto do joelho direito",
        "recorde de ângulo do joelho direito"
    ],

    "menor_angulo_joelho_direito": [
        "menor ângulo do joelho direito",
        "mínimo do joelho R",
        "qual foi o menor valor no joelho direito",
        "joelho R chegou até qual mínimo?",
        "qual menor grau do joelho direito",
        "qual valor mais baixo do joelho direito",
        "me diz o menor ângulo do joelho direito",
        "qual a menor abertura do joelho direito?",
        "piso de flexão do joelho direito",
        "mínimo do joelho direito",
        "menor grau do joelho direito",
        "o joelho direito atingiu qual mínimo?",
        "qual foi o ângulo mínimo do joelho direito?"
    ],

    "media_angulo_ombro_direito": [
        "qual a média do ombro direito",
        "média ângulo ombro R",
        "média do ombro direito",
        "qual foi a média do ombro R",
        "média dos movimentos do ombro direito",
        "qual o valor médio do ombro R",
        "ombro direito teve qual média de ângulos?",
        "me informa a média do ângulo do ombro direito",
        "qual a média geral do ombro direito?",
        "média de movimento do ombro R",
        "qual a média de flexão do ombro direito?",
        "o ombro direito teve qual valor médio?",
        "média do ângulo do ombro direito"
    ],

    "saudacao": [
        "oi",
        "olá",
        "bom dia",
        "boa tarde",
        "boa noite",
        "e aí",
        "fala",
        "salve",
        "opa",
        "tudo bem?",
        "como vai?",
        "iae",
        "alô",
        "oi tudo bem?",
        "fala aí",
        "e aí beleza?"
    ],

    "encerramento": [
        "tchau",
        "obrigado",
        "valeu",
        "até mais",
        "até logo",
        "flw",
        "até a próxima",
        "encerra aí",
        "abraços",
        "fui",
        "vlw",
        "obrigado pela ajuda",
        "agradeço",
        "até breve",
        "falou",
        "foi bom conversar",
        "encerra a conversa"
    ]
}
texts = []
labels = []
for label, exemplos in intents.items():
    for ex in exemplos:
        texts.append(ex)
        labels.append(label)

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

pipeline_model = Pipeline([("bert_vec", BERTVectorizer(tokenizer, model)), ("clf", LogisticRegression(max_iter=1000))])
pipeline_model.fit(texts, y_encoded)

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
        if intent == "maior_angulo_joelho_esquerdo":
            valor = df['kneeLangle'].max()
            return f"O maior ângulo do joelho esquerdo é {valor:.2f}º."
        elif intent == "menor_angulo_joelho_esquerdo":
            valor = df['kneeLangle'].min()
            return f"O menor ângulo do joelho esquerdo é {valor:.2f}º."
        elif intent == "maior_angulo_joelho_direito":
            valor = df['kneeRangle'].max()
            return f"O maior ângulo do joelho direito é {valor:.2f}º."
        elif intent == "menor_angulo_joelho_direito":
            valor = df['kneeRangle'].min()
            return f"O menor ângulo do joelho direito é {valor:.2f}º."

        # OMBRO
        elif intent == "maior_angulo_ombro_esquerdo":
            valor = df['shoulderLangle'].max()
            return f"O maior ângulo do ombro esquerdo é {valor:.2f}º."
        elif intent == "menor_angulo_ombro_esquerdo":
            valor = df['shoulderLangle'].min()
            return f"O menor ângulo do ombro esquerdo é {valor:.2f}º."
        elif intent == "maior_angulo_ombro_direito":
            valor = df['shoulderRangle'].max()
            return f"O maior ângulo do ombro direito é {valor:.2f}º."
        elif intent == "menor_angulo_ombro_direito":
            valor = df['shoulderRangle'].min()
            return f"O menor ângulo do ombro direito é {valor:.2f}º."

        # COTOVELO
        elif intent == "maior_angulo_cotovelo_esquerdo":
            valor = df['elbowLangle'].max()
            return f"O maior ângulo do cotovelo esquerdo é {valor:.2f}º."
        elif intent == "menor_angulo_cotovelo_esquerdo":
            valor = df['elbowLangle'].min()
            return f"O menor ângulo do cotovelo esquerdo é {valor:.2f}º."
        elif intent == "maior_angulo_cotovelo_direito":
            valor = df['elbowRangle'].max()
            return f"O maior ângulo do cotovelo direito é {valor:.2f}º."
        elif intent == "menor_angulo_cotovelo_direito":
            valor = df['elbowRangle'].min()
            return f"O menor ângulo do cotovelo direito é {valor:.2f}º."

        # OUTROS
        elif intent == "saudacao":
            if name_patient:
                return random.choice([
                    f"Olá {username}! Posso ajudar na análise dos movimentos do paciente {name_patient}",
                    f"Oi {username}! Pode me perguntar sobre máximos, mínimos e média dos ângulos.",
                    f"Eai {username}! Como posso ajudar com o documento de {name_patient}?"
                ])
            else:
                return random.choice([
                    f"Olá {username}! Posso ajudar na análise dos movimentos do paciente!",
                    f"Oi {username}! Pode me perguntar sobre máximos, mínimos e média dos ângulos.",
                    f"Eai {username}! Como posso ajudar com o documento?"
                ])
        elif intent == "encerramento":
            return random.choice([
                f"Até logo {username}! Qualquer coisa é só chamar.",
                f"Encerrando por aqui.",
                f"Tchau {username}! Estou à disposição se precisar novamente.",
                f"Foi um prazer ajudar. {username}. Até a próxima!",
                f"Encerrando a análise do paciente {name_patient}. Até mais {username}!"
            ])
        else:
            return f"Desculpe {username}, não entendi"
    else:
        return f"{username.captilize()}, você precisa fazer upload do arquivo CSV/XLSX primeiro! ☝️"
