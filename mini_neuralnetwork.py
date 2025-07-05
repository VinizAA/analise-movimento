from collections import defaultdict
import random, unicodedata, string
import streamlit as st
import pandas as pd
import io, re

from supabase import create_client

# Supabase config
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def contains_word(word, text):
    return re.search(rf"\b{re.escape(word)}\b", text) is not None

def clean_text(text):
    substitutions = {
        "vc": "voce",
        "vcs": "voces",
        "c": "voce",
        "td": "tudo",
        "q": "que",
        "pq": "porque",
        "pqc": "por que",
        "oq": "o que",
        "kd": "cadê",
        "tb": "tambem",
        "tbm": "tambem",
        "blz": "beleza",
        "vlw": "valeu",
        "flw": "falou",
        "obg": "obrigado",
        "dps": "depois",
        "msg": "mensagem",
        "hj": "hoje",
        "bjs": "beijos",
        "fds": "fim de semana",
        "pfv": "por favor",
        "pls": "por favor",
        "agr": "agora",

        "eh": "é",
        "ta": "esta",
        "tá": "esta",
        "to": "estou",
        "tô": "estou",
        "tamo": "estamos",
        "vamo": "vamos",
        "num": "nao",
        "n": "nao",
        "naum": "nao",
        "ñ": "nao",
        "mt": "muito",
        "mto": "muito",
    }

    text_format = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text_clean = text_format.translate(str.maketrans('', '', string.punctuation)).lower()

    words = text_clean.split()
    normalized_words = [substitutions.get(word, word) for word in words]

    return " ".join(normalized_words).strip()

def identify_art(prompt_user):
    if "joelho direito" in prompt_user:
        return "kneeRangle"
    elif "joelho esquerdo" in prompt_user:
        return "kneeLangle"
    elif "cotovelo direito" in prompt_user:
        return "elbowRangle"
    elif "cotovelo esquerdo" in prompt_user:
        return "elbowLangle"
    elif "ombro direito" in prompt_user:
        return "shoulderRangle"
    elif "ombro esquerdo" in prompt_user:
        return "shoulderLangle"
    else:
        return None
    
def chatbot_brain(prompt_user, file_patient, file_training, name_patient):
    phrases_per_section = defaultdict(list)
    df = None
    
    if isinstance(file_patient, str):
        try:
            doc = supabase.storage.from_("pacientes").download(file_patient)
            text_data = io.StringIO(doc.decode("utf-8"))
            df = pd.read_csv(text_data)
        except Exception as e:
            return f"❌ Erro ao carregar documento do paciente {name_patient}: {e}"
    elif isinstance(file_patient, pd.DataFrame):
        df = file_patient

    try:
        with open(file_training, "r", encoding="utf-8") as f:
            current_section = None
            for line in f:
                line = line.strip().lower()
                if not line:
                    continue
                if line.startswith("---") and line.endswith("---"):
                    current_section = line.strip("-")
                elif current_section:
                    phrases_per_section[current_section].append(line)
    except Exception as e:
        return f"❌ Erro ao carregar o arquivo de treinamento: {e}"

    if df:
        prompt_user = clean_text(prompt_user)
        column_name = identify_art(prompt_user)

        if name_patient:
            answers = {
                "saudacao": [
                    f"Olá! Me pergunte sobre o documento de {name_patient}!",
                    f"Eaí, Estou pronto para analisar o documento de {name_patient}. Pergunte algo!",
                    f"Opa! O que você deseja saber sobre o documento de {name_patient}?",
                    f"Pode me perguntar qualquer coisa sobre o documento do paciente {name_patient}!"
                ],
                "encerramento": [
                    "Valeu!", "Até mais!", "Tchau!", "Tamo junto!"
                ],
                "ajuda": [
                    "Pode me perguntar sobre o maior, menor ou média de ângulo de uma articulação",
                    "Você pode me perguntar sobre os ângulos de uma articulação",
                ]
            }
        else:
            answers = {
                "saudacao": [
                    f"Olá! Me pergunte sobre o documento do paciete!",
                    f"Eaí, Estou pronto para analisar o documento do paciente. Pergunte algo!",
                    f"Opa! O que você deseja saber sobre o documento do paciente?",
                    f"Pode me perguntar qualquer coisa sobre o documento do paciente!"
                ],
                "encerramento": [
                    "Valeu!", "Até mais!", "Tchau!", "Tamo junto!"
                ],
                "ajuda": [
                    "Pode me perguntar sobre o maior, menor ou média de ângulo de uma articulação",
                    "Você pode me perguntar sobre os ângulos de uma articulação",
                ]
            }

        if column_name != None:
            if column_name == "kneeLangle":
                column_name_format = "joelho esquerdo"
            elif column_name == "kneeRangle":
                column_name_format = "joelho direito"
            elif column_name == "elbowLangle":
                column_name_format = "cotovelo esquerdo"
            elif column_name == "elbowRangle":
                column_name_format = "cotovelo direito"
            elif column_name == "shoulderLangle":
                column_name_format = "ombro esquerdo"
            elif column_name == "shoulderRangle":
                column_name_format = "ombro direito"
            else:
                column_name_format = column_name

            max_val = df[column_name].max()
            min_val = df[column_name].min()
            med_val = df[column_name].mean()

            answers.update({
                "maior_angulo": [f"O maior ângulo do {column_name_format} é **{max_val:.2f}**!"],
                "menor_angulo": [f"O menor ângulo do {column_name_format} é **{min_val:.2f}**!"],
                "media_angulo": [f"A média dos ângulos do {column_name_format} é **{med_val:.2f}**!"]
            })

        for category in ["saudacao", "encerramento", "ajuda", "maior_angulo", "menor_angulo", "media_angulo"]:
            for word_train in phrases_per_section[category]:
                if contains_word(clean_text(word_train), prompt_user):
                    return random.choice(answers[category])

        return "Desculpe, não entendi"
    else:
        return "Faça upload do arquivo CSV primeiro! ☝️"