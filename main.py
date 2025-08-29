# app.py
import os
import sqlite3
import hashlib
import time
import re
import unicodedata
import plotly
import json

from flask import (Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONFIGURAÇÃO DO FLASK
# ==============================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'uma-chave-secreta-muito-forte-e-dificil-de-adivinhar'
app.config['UPLOAD_FOLDER'] = 'docs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_PATH = "datai.db"

# ==============================
# BANCO DE DADOS (SQLite)
# ==============================
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cursor = db.cursor()
    # Tabela de usuários com hash de senha
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash TEXT NOT NULL
    )
    """)
    # Tabela de pacientes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pacientes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        sobrenome TEXT,
        sexo TEXT,
        data_nascimento TEXT,
        idade INTEGER,
        documento_url TEXT,
        user_id TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    db.commit()

# Roda a inicialização do DB uma vez ao iniciar
with app.app_context():
    init_db()

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def get_current_user_id():
    return session.get('user_id')

def verify_user_access(paciente_id, user_id):
    db = get_db()
    paciente = db.execute("SELECT id FROM pacientes WHERE id = ? AND user_id = ?",
                          (paciente_id, user_id)).fetchone()
    return paciente is not None

def slugify(value):
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^a-zA-Z0-9_-]', '_', value)
    return value

def extract_original_name(path):
    # Extrai o nome original do arquivo salvo (após o timestamp)
    try:
        return '_'.join(os.path.basename(path).split('_')[3:])
    except IndexError:
        return os.path.basename(path)

# ==============================
# ROTAS DE AUTENTICAÇÃO
# ==============================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action')
        db = get_db()

        if action == 'login':
            username = request.form['username']
            password = request.form['password']
            user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

            if user and check_password_hash(user['password_hash'], password):
                session.clear()
                session['user_id'] = user['id']
                session['username'] = user['username']
                return redirect(url_for('dashboard'))
            else:
                flash('Usuário ou senha inválidos.', 'error')

        elif action == 'register':
            username = request.form['new_username']
            password = request.form['new_password']
            user_exists = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()

            if user_exists:
                flash('Esse usuário já existe.', 'error')
            else:
                user_id = hashlib.md5(username.encode()).hexdigest()
                password_hash = generate_password_hash(password)
                db.execute(
                    "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
                    (user_id, username, f"{username}@datai.local", password_hash)
                )
                db.commit()
                flash('Conta criada com sucesso! Faça o login.', 'success')
                return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Você saiu da sua conta.', 'success')
    return redirect(url_for('login'))

# ==============================
# ROTAS DA APLICAÇÃO PRINCIPAL
# ==============================
@app.route('/')
def dashboard():
    user_id = get_current_user_id()
    if not user_id:
        return redirect(url_for('login'))

    db = get_db()
    pacientes = db.execute("SELECT id, nome, sobrenome FROM pacientes WHERE user_id = ?", (user_id,)).fetchall()
    return render_template('dashboard.html', pacientes=pacientes)


@app.route('/pacientes/adicionar', methods=['GET', 'POST'])
def create_paciente():
    user_id = get_current_user_id()
    if not user_id:
        return redirect(url_for('login'))

    if request.method == 'POST':
        nome = request.form['nome']
        sobrenome = request.form['sobrenome']
        sexo = request.form['sexo']
        date_nasc_str = request.form['data_nascimento']
        documentos = request.files.getlist('documentos')

        if not all([nome, sobrenome, sexo, date_nasc_str]) or not documentos or documentos[0].filename == '':
            flash('Preencha todos os campos e selecione pelo menos um documento.', 'warning')
            return redirect(request.url)

        date_nasc = datetime.strptime(date_nasc_str, '%Y-%m-%d').date()
        idade = date.today().year - date_nasc.year - ((date.today().month, date.today().day) < (date_nasc.month, date_nasc.day))

        documentos_urls = []
        for doc in documentos:
            filename = f"user_{user_id}_{int(time.time())}_{secure_filename(doc.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            doc.save(path)
            documentos_urls.append(path)

        db = get_db()
        db.execute("""
            INSERT INTO pacientes (nome, sobrenome, sexo, data_nascimento, idade, documento_url, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (nome, sobrenome, sexo, date_nasc.isoformat(), idade, ",".join(documentos_urls), user_id))
        db.commit()
        flash('Paciente cadastrado com sucesso!', 'success')
        return redirect(url_for('dashboard'))

    min_date = date.today() - relativedelta(years=120)
    max_date = date.today()
    return render_template('add_patient.html', min_date=min_date, max_date=max_date)

@app.route('/pacientes/editar/<int:paciente_id>', methods=['GET', 'POST'])
def edit_patient(paciente_id):
    user_id = get_current_user_id()
    if not user_id or not verify_user_access(paciente_id, user_id):
        flash('Acesso negado.', 'error')
        return redirect(url_for('dashboard'))

    db = get_db()
    paciente = db.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()

    if request.method == 'POST':
        nome = request.form['nome']
        sobrenome = request.form['sobrenome']
        sexo = request.form['sexo']
        date_nasc_str = request.form['data_nascimento']
        date_nasc_new = datetime.strptime(date_nasc_str, '%Y-%m-%d').date()
        novos_docs = request.files.getlist('novos_documentos')

        novos_urls = paciente['documento_url'].split(',') if paciente['documento_url'] else []
        if novos_docs and novos_docs[0].filename != '':
            # Apaga arquivos antigos se desejar ou apenas adiciona. Aqui substituímos.
            novos_urls = []
            for doc in novos_docs:
                filename = f"user_{user_id}_{int(time.time())}_{secure_filename(doc.filename)}"
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                doc.save(path)
                novos_urls.append(path)

        idade = date.today().year - date_nasc_new.year - ((date.today().month, date.today().day) < (date_nasc_new.month, date_nasc_new.day))
        
        db.execute("""
            UPDATE pacientes SET nome=?, sobrenome=?, sexo=?, data_nascimento=?, idade=?, documento_url=?
            WHERE id=? AND user_id=?
        """, (nome, sobrenome, sexo, date_nasc_new.isoformat(), idade, ",".join(novos_urls), paciente_id, user_id))
        db.commit()
        flash('Paciente atualizado com sucesso!', 'success')
        return redirect(url_for('dashboard'))

    current_docs = [(url, extract_original_name(url)) for url in paciente['documento_url'].split(',')]
    return render_template('edit_patient.html', paciente=paciente, current_docs=current_docs)


@app.route('/pacientes/excluir/<int:paciente_id>', methods=['POST'])
def delete_patient(paciente_id):
    user_id = get_current_user_id()
    if not user_id or not verify_user_access(paciente_id, user_id):
        flash('Acesso negado.', 'error')
        return redirect(url_for('dashboard'))

    db = get_db()
    # Opcional: excluir arquivos do disco
    paciente = db.execute("SELECT documento_url FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
    for url in paciente['documento_url'].split(','):
        if os.path.exists(url):
            os.remove(url)

    db.execute("DELETE FROM pacientes WHERE id = ? AND user_id = ?", (paciente_id, user_id))
    db.commit()
    flash('Paciente excluído com sucesso!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/pacientes/analise/<int:paciente_id>')
def view_patient(paciente_id):
    user_id = get_current_user_id()
    
    if not user_id or not verify_user_access(paciente_id, user_id):
        flash('Acesso negado.', 'error')
        return redirect(url_for('dashboard'))

    db = get_db()
    paciente = db.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()

    if not paciente or not paciente['documento_url']:
        flash('Paciente não encontrado ou sem documentos.', 'error')
        return redirect(url_for('dashboard'))

    first_doc_path = paciente['documento_url'].split(',')[0]

    try:
        if first_doc_path.lower().endswith(".csv"):
            data = pd.read_csv(first_doc_path, dtype=str, decimal=',')
        else:
            data = pd.read_excel(first_doc_path, engine='openpyxl', dtype=str)

        data.columns = data.columns.str.strip()

        preferred = ["Goniometry UpLeg Angle", "Goniometry Leg Angle"]
        angle_cols = [c for c in preferred if c in data.columns]

        if not angle_cols:
            angle_cols = [c for c in data.columns if "angle" in c.lower() or "goniometry" in c.lower()]

        time_candidates = ["Time", "time", "Timestamp", "timestamp", "Tempo", "tempo"]
        time_col = next((c for c in time_candidates if c in data.columns), None)

        if not time_col or not angle_cols:
            flash(f'Arquivo do paciente faltando colunas necessárias (Time/Angle). Encontradas: {data.columns.tolist()}', 'error')
            return redirect(url_for('dashboard'))

        def to_num(series):
            return pd.to_numeric(series.astype(str).str.replace(',', '.').str.strip(), errors='coerce')

        data[time_col] = to_num(data[time_col])
        for c in angle_cols:
            data[c] = to_num(data[c])

        data = data.dropna(subset=[time_col])
        if data.empty:
            flash('Arquivo não possui valores válidos na coluna de tempo.', 'error')
            return redirect(url_for('dashboard'))

        valid_angle_cols = [c for c in angle_cols if data[c].notna().any()]
        if not valid_angle_cols:
            flash('Nenhuma coluna de ângulo com valores válidos encontrada.', 'error')
            return redirect(url_for('dashboard'))

        intuitive_names = {
            "Goniometry UpLeg Angle": "Coxa",
            "Goniometry Leg Angle": "Canela"
        }

        # Estatísticas básicas
        stats = []
        for c in valid_angle_cols:
            stats.append({
                'name': intuitive_names.get(c, c),
                'min': f"{str(data[c].min()).replace('.', ',')}°" if pd.notna(data[c].min()) else "-",
                'max': f"{str(data[c].max()).replace('.', ',')}°" if pd.notna(data[c].max()) else "-",
                'mean': f"{str(data[c].mean().round(1)).replace('.', ',')}°" if pd.notna(data[c].mean()) else "-"
            })

        # Gráfico
        fig = go.Figure()
        for c in valid_angle_cols:
            fig.add_trace(go.Scatter(
                x=data[time_col].tolist(),
                y=data[c].tolist(),
                mode="lines",
                name=intuitive_names.get(c, c)
            ))
        fig.update_layout(
            title="Ângulos Articulares",
            xaxis_title="Tempo",
            yaxis_title="Ângulo",
            autosize=True
        )

        graphJSON_angles = fig.to_dict()

    except FileNotFoundError:
        flash('Arquivo do paciente não encontrado no servidor.', 'error')
        return redirect(url_for('dashboard'))

    return render_template(
        'view_patient.html',
        paciente=paciente,
        stats=stats,
        graph_angle=graphJSON_angles,
    )

# ==============================
# ROTA DE CONVIDADO
# ==============================
@app.route('/guest', methods=['GET', 'POST'])
def guest_analysis():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if not uploaded_file or uploaded_file.filename == '':
            flash('Por favor, selecione um arquivo.', 'warning')
            return redirect(request.url)

        try:
            if uploaded_file.filename.lower().endswith(".csv"):
                uploaded_file.seek(0)
                data = pd.read_csv(uploaded_file, dtype=str)
            elif uploaded_file.filename.lower().endswith((".xlsx", ".xls")):
                uploaded_file.seek(0)
                data = pd.read_excel(uploaded_file, engine='openpyxl', dtype=str)
            else:
                flash('Formato de arquivo inválido. Use .csv ou .xlsx.', 'error')
                return redirect(request.url)
            data.columns = data.columns.str.strip()

            def robust_to_num(series):
                if series.str.contains(',', na=False).any():
                    cleaned_series = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                else:
                    cleaned_series = series
                return pd.to_numeric(cleaned_series.str.strip(), errors='coerce')

            angle_cols = [c for c in data.columns if "angle" in c.lower() or "goniometry" in c.lower()]
            time_candidates = ["Time", "time", "Timestamp", "timestamp", "Tempo", "tempo"]
            time_col = next((c for c in time_candidates if c in data.columns), None)

            if not time_col or not angle_cols:
                flash(f'Arquivo não contém colunas de Tempo ou Ângulo.', 'error')
                return redirect(request.url)

            data[time_col] = robust_to_num(data[time_col])
            for c in angle_cols:
                data[c] = robust_to_num(data[c])

            data_for_graph = data.dropna(subset=[time_col])

            valid_angle_cols = [c for c in angle_cols if data[c].notna().any()]

            if not valid_angle_cols:
                flash('Nenhuma coluna de ângulo com dados válidos foi encontrada.', 'warning')
                return redirect(request.url)

            intuitive_names = {"Goniometry UpLeg Angle": "Coxa", "Goniometry Leg Angle": "Canela"}
            stats = []
            for c in valid_angle_cols:
                stats.append({
                    'name': intuitive_names.get(c, c.replace('_', ' ').title()),
                    'min': data[c].min(), 'max': data[c].max(), 'mean': data[c].mean(),
                })

            fig = go.Figure()
            for c in valid_angle_cols:
                fig.add_trace(go.Scatter(
                    x=data_for_graph[time_col].tolist(),
                    y=data_for_graph[c].tolist(), 
                    mode="lines",
                    name=intuitive_names.get(c, c.replace('_', ' ').title())
                ))
            graphJSON_angle = fig.to_dict()

            print(graphJSON_angle)
            return render_template('guest.html', stats=stats, graph_angle=graphJSON_angle)

        except Exception as e:
            flash(f'Ocorreu um erro interno ao processar o arquivo: {e}', 'error')
            return redirect(request.url)

    return render_template('guest.html')

# Rota para servir os logos (opcional, pode-se usar /static/img/...)
@app.route('/logos/<filename>')
def logos(filename):
    return send_from_directory('static/img', filename)

if __name__ == '__main__':
    app.run(debug=True)