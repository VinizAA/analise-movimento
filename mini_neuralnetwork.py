import re
import json
import random
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from rapidfuzz import process, fuzz
from spellchecker import SpellChecker

spell = SpellChecker(language='pt')

def corrigir_texto(texto):
    palavras = texto.split()
    palavras_corrigidas = [spell.correction(p) for p in palavras]
    return ' '.join(palavras_corrigidas)

def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

with open("intent_data.json", "r", encoding="utf-8") as f:
    intent_data = json.load(f)

with open("phrases.json", "r", encoding="utf-8") as f:
    phrases_data = json.load(f)
    saudacoes = phrases_data.get("saudacoes", [])
    encerramentos = phrases_data.get("encerramentos", [])

samples = []
intents = []

for label, frases in intent_data.items():
    for frase in frases:
        frase_norm = normalize_text(frase)
        samples.append(frase_norm)
        intents.append(label)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(samples)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, intents)

def predict(question):
    question_format = corrigir_texto(normalize_text(question))

    x = vectorizer.transform([question_format])
    return clf.predict(x)[0]

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

    all_terms = []
    key_for_term = {}
    for key, terms in mapping.items():
        for term in terms:
            all_terms.append(term)
            key_for_term[term] = key

    match, score, _ = process.extractOne(texto, all_terms, scorer=fuzz.partial_ratio)
    if score >= 80:
        print(key_for_term[match])
        return key_for_term[match]
    else:
        return None

def answer(df, question):
    intent = predict(question)
    art = extract(question)

    nomes = {
        "kneeLangle": "Joelho Esquerdo",
        "kneeRangle": "Joelho Direito",
        "elbowLangle": "Cotovelo Esquerdo",
        "elbowRangle": "Cotovelo Direito",
        "shoulderLangle": "Ombro Esquerdo",
        "shoulderRangle": "Ombro Direito"
    }

    print("Frase:", question)
    print("Intent detectada:", predict(question))
    print("Articulação detectada:", extract(question))

    if intent == "saudacao":
        return {"text": random.choice(saudacoes)}

    if intent == "encerramento":
        return {"text": random.choice(encerramentos)}

    if intent in ["maior_angulo", "menor_angulo", "media_angulo", "amplitude_angulo"]:
        if art is None:
            return "Você deseja o valor de qual articulação?"
        if art not in df.columns:
            return f"Desculpe, a articulação '{nomes.get(art, art)}' não está presente nos dados."

        if intent == "maior_angulo":
            val = df[art].max()
            return f"O maior ângulo do {nomes[art]} é **{val:.1f}°**."
        elif intent == "menor_angulo":
            val = df[art].min()
            return f"O menor ângulo do {nomes[art]} é **{val:.1f}°**."
        elif intent == "media_angulo":
            val = df[art].mean()
            return f"A média dos ângulos do {nomes[art]} é **{val:.1f}°**."
        elif intent == "amplitude_angulo":
            val = df[art].max() - df[art].min()
            return f"A amplitude de movimento do {nomes[art]} é **{val:.1f}°**."

    return "Desculpe, não entendi sua pergunta."

