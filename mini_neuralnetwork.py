from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import re, random, json, unicodedata

def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

with open("intent_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("phrases.json", "r", encoding="utf-8") as f:
    phrases_data = json.load(f)
    saudacoes = phrases_data["saudacoes"]
    encerramentos = phrases_data["encerramentos"]

samples, intents = [], []
for label, phrases in data.items():
    phrases_norm = [normalize_text(frase) for frase in phrases]
    samples.extend(phrases_norm)
    intents.extend([label] * len(phrases_norm))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(samples)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, intents)

def predict(question):
    question = normalize_text(question)
    x = vectorizer.transform([question])
    return knn.predict(x)[0]

def extract(texto):
    mapping = {
        "kneeLangle": ["joelho esquerdo", "knee left"],
        "kneeRangle": ["joelho direito", "knee right"],
        "elbowLangle": ["cotovelo esquerdo", "elbow left"],
        "elbowRangle": ["cotovelo direito", "elbow right"],
        "shoulderLangle": ["ombro esquerdo", "shoulder left"],
        "shoulderRangle": ["ombro direito", "shoulder right"],
    }

    texto = texto.lower()
    for key, terms in mapping.items():
        for term in terms:
            if term in texto:
                return key
    return None

def answer(df, question):
    intent = predict(question) 
    art = extract(question)

    names = {
        "kneeLangle": "Joelho Esquerdo",
        "kneeRangle": "Joelho Direito",
        "elbowLangle": "Cotovelo Esquerdo",
        "elbowRangle": "Cotovelo Direito",
        "shoulderLangle": "Ombro Esquerdo",
        "shoulderRangle": "Ombro Direito",
    }
    
    if intent == "saudacao":
        return {
            "text": random.choice(saudacoes)
        }
    
    if intent == "encerramento":
        return {
            "text": random.choice(encerramentos)
        }

    if art is None:
        if intent in ["maior_angulo", "menor_angulo", "media_angulo", "amplitude_angulo"]:
            return "Você deseja o valor de qual articulação?"
        return "Desculpe, não consegui entender"

    if art not in df.columns:
        return f"Desculpe, a articulação '{names.get(art, art)}' não está presente nos dados."

    if intent == "maior_angulo":
        val = df[art].max()
        return f"O maior ângulo do {names[art]} é **{val:.1f}°**."

    elif intent == "menor_angulo":
        val = df[art].min()
        return f"O menor ângulo do {names[art]} é **{val:.1f}°**."

    elif intent == "media_angulo":
        val = df[art].mean()
        return f"A média dos ângulos do {names[art]} é **{val:.1f}°**."
    
    elif intent == "amplitude_angulo":
        val = df[art].max() - df[art].min()
        return f"A amplitude de movimento do {names[art]} é **{val:.1f}°**."

    return "Desculpe, não entendi sua pergunta."