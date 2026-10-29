# 📘 MANUAL TÉCNICO Y GUÍA INTEGRAL DE OPERACIÓN
# Sistema Clínico Ch'aweb'al (K'iche' ⇄ Español)

---

## 📋 FICHA TÉCNICA DEL SISTEMA

| Parámetro | Especificación |
| :--- | :--- |
| **Nombre del Sistema** | **Ch'aweb'al** *(Traductor Clínico y Plataforma de IA K'iche' - Español)* |
| **Versión Actual** | `2.1.0-Production` |
| **Área de Aplicación** | Salud Pública, Centros de Atención Primaria, Triaje Médico Intercultural |
| **Lenguajes Soportados** | K'iche' (*quc / Variante Central / ALMG*) y Español (*es-GT / es-419*) |
| **Lenguaje de Programación** | Python 3.9+ / JavaScript (ES6+) |
| **Framework Backend** | Flask 3.x / Flask-SQLAlchemy / Gunicorn |
| **Motor de IA / ASR** | Meta MMS (Wav2Vec2 CTC - `facebook/mms-1b-all`) con Adaptador LoRA PEFT |
| **Motor ASR Español** | OpenAI Whisper (vía Hugging Face Transformers) |
| **Síntesis de Voz (TTS)** | Coqui TTS Custom K'iche' (Synthesizer FP32) + Caché de Audio Nativo + gTTS |
| **Traducción Lingüística** | Diccionario Médico Ponderado + Búsqueda Difusa (*RapidFuzz*) + Fallback Neuronal |
| **IA Generativa Complementaria** | Google Gemini Pro API (Generación y revisión de terminología médica) |
| **Base de Datos** | SQLite 3 (ORM SQLAlchemy) con migración dinámica de esquemas |
| **Almacenamiento en la Nube** | Amazon Web Services (AWS S3) para datasets y respaldos |

---

## 📑 ÍNDICE DE CONTENIDOS

1. [Arquitectura Global del Sistema](#1-arquitectura-global-del-sistema)
2. [Requisitos e Infraestructura](#2-requisitos-e-infraestructura)
3. [Instalación y Puesta en Marcha](#3-instalación-y-puesta-en-marcha)
4. [Seguridad, Base de Datos y Control de Acceso (RBAC)](#4-seguridad-base-de-datos-y-control-de-acceso-rbac)
5. [Guía de Operación Módulo por Módulo](#5-guía-de-operación-módulo-por-módulo)
   - [5.1 Módulo 1: Acceso, Registro y Recuperación de Cuenta](#51-módulo-1-acceso-registro-y-recuperación-de-cuenta)
   - [5.2 Módulo 2: Traductor Clínico Bidireccional](#52-módulo-2-traductor-clínico-bidireccional)
   - [5.3 Módulo 3: Preferencias, Rendimiento y Cuenta](#53-módulo-3-preferencias-rendimiento-y-cuenta)
   - [5.4 Módulo 4: Entrenamiento de Voz y Expansión de Datasets](#54-módulo-4-entrenamiento-de-voz-y-expansión-de-datasets)
   - [5.5 Módulo 5: Práctica de Escritura y Revisión Lingüística](#55-módulo-5-práctica-de-escritura-y-revisión-lingüística)
   - [5.6 Módulo 6: Panel de Administración y Auditoría](#56-módulo-6-panel-de-administración-y-auditoría)
6. [Referencia de la API REST](#6-referencia-de-la-api-rest)
7. [Motor Lingüístico y Diccionario Médico](#7-motor-lingüístico-y-diccionario-médico)
8. [Resolución de Problemas (Troubleshooting) y Mantenimiento](#8-resolución-de-problemas-troubleshooting-y-mantenimiento)

---

# 1. ARQUITECTURA GLOBAL DEL SISTEMA

El sistema **Ch'aweb'al** está diseñado bajo una arquitectura de microservicios embebidos orientada a entornos clínicos con conectividad variable. Integra modelos de aprendizaje profundo (*Deep Learning*) para procesamiento de señales de audio con un sistema transaccional web ligero.

### Diagrama de Arquitectura de Componentes

```mermaid
graph TB
    subgraph Cliente ["Capa de Cliente (Navegador Web / Dispositivo Móvil)"]
        UI["Interfaz Web Responsiva (HTML5 / Tailwind CSS / Glassmorphism)"]
        RecAudio["Grabador Web Audio API (16kHz / WebM / WAV)"]
        Anatomy["Modelo Anatómico SVG Interactivo"]
        AudioOut["Reproductor HTML5 Audio"]
    end

    subgraph Servidor ["Capa de Aplicación (Flask / Gunicorn)"]
        Router["Enrutador de Rutas & Middleware RBAC"]
        AuthModule["Módulo de Autenticación & PBKDF2"]
        SummaryEngine["Generador de Resumen Clínico"]
        ExportJob["Gestor de Tareas Asíncronas (Export S3)"]
    end

    subgraph IAPipeline ["Capa de Inteligencia Artificial & Procesamiento de Señales"]
        PreProc["Preprocesador: DC-Offset + Peak Normalization + Trim Librosa"]
        ASR_Kiche["ASR K'iche': Meta MMS-1B Wav2Vec2 + LoRA PEFT (quc)"]
        ASR_ES["ASR Español: Whisper Pipeline"]
        NormEngine["Motor de Normalización ALMG & Glotales"]
        FuzzyMatcher["Búsqueda Difusa RapidFuzz (Ratio >= 88)"]
        DictEngine["Diccionario Médico Jerárquico (5 Niveles)"]
        TTS_Custom["Coqui TTS Synthesizer (FP32 K'iche')"]
        TTS_Cache["Caché de Audio Nativo de Hablantes"]
        TTS_ES["gTTS (Google Text-to-Speech Español)"]
        GeminiAPI["Google Gemini Pro (IA Generativa Externa)"]
    end

    subgraph Persistencia ["Capa de Persistencia & Cloud"]
        DB[(SQLite app.db: Users & Audit)]
        JSONData[medical_dictionary.json & normalization_rules.json]
        CSVData[escritura_dataset.csv & training_data.csv]
        S3Storage[Amazon AWS S3 Bucket]
    end

    UI --> Router
    RecAudio --> Router
    Router --> AuthModule
    AuthModule --> DB
    Router --> PreProc
    PreProc --> ASR_Kiche
    PreProc --> ASR_ES
    ASR_Kiche --> NormEngine
    NormEngine --> FuzzyMatcher
    FuzzyMatcher --> DictEngine
    DictEngine --> JSONData
    DictEngine --> CSVData
    DictEngine --> TTS_Cache
    DictEngine --> TTS_Custom
    Router --> SummaryEngine
    Router --> ExportJob
    ExportJob --> S3Storage
    Router --> GeminiAPI
```

---

# 2. REQUISITOS E INFRAESTRUCTURA

### 2.1 Requisitos de Hardware

| Entorno | Procesador | Memoria RAM | Almacenamiento | Aceleración Gráfica |
| :--- | :--- | :--- | :--- | :--- |
| **Desarrollo Local** | Intel i5 / AMD Ryzen 5 / Apple Silicon (M1/M2/M3) | 8 GB mínimo (16 GB recomendado) | 10 GB libres | Opcional (CUDA 11.8+ o MPS en macOS) |
| **Servidor en la Nube (AWS EC2 / VPS)** | 1 vCPU (ej. `t2.micro` o `t3.small`) | 1 GB a 2 GB RAM físicos | 20 GB SSD | **Obligatorio crear 4 GB de Swap** |
| **Servidor con GPU dedicada** | 4 vCPU | 16 GB RAM | 50 GB NVMe | NVIDIA T4 / A10G / RTX 3060+ (8GB+ VRAM) |

> [!IMPORTANT]
> En servidores gratuitos de AWS EC2 (`t2.micro` con 1 GB de RAM), es **estrictamente indispensable** habilitar un archivo Swap de al menos 4 GB para evitar que el compilador de grafos de PyTorch sea terminado por el *Out-Of-Memory Killer (OOM)* del kernel Linux durante la inferencia.

### 2.2 Requisitos de Software y Dependencias de Sistema

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    build-essential \
    espeak-ng \
    git \
    nginx \
    certbot \
    python3-certbot-nginx
```

---

# 3. INSTALACIÓN Y PUESTA EN MARCHA

### 3.1 Clonación y Configuración del Entorno Virtual

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO> traductorKiche-Esp
cd traductorKiche-Esp/traductor_kiche

# 2. Crear y activar entorno virtual Python
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Actualizar el gestor de paquetes e instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Archivo de Variables de Entorno (`.env`)

Cree un archivo `.env` en la raíz de `traductor_kiche/` con los siguientes parámetros:

```ini
# Configuración del Servidor Flask
SECRET_KEY=c34a9b5f928e1928374a81d002fbc947
FLASK_ENV=production
PORT=5000

# AWS S3 Cloud Sync (Opcional para persistencia externa de datasets)
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=traductor-kiche-datasets-bucket

# Token para servicios de IA Complementaria
GEMINI_API_KEY=AIzaSyD_EXAMPLE_KEY_GEMINI_PRO
HF_TOKEN=hf_EXAMPLE_TOKEN_HUGGINGFACE
```

### 3.3 Inicialización y Ejecución Local

#### En macOS / Linux:
```bash
chmod +x run_web_app.sh
./run_web_app.sh
```

#### En Windows:
```cmd
run_web_app.bat
```

La consola confirmará el inicio y descargará automáticamente los pesos de Meta MMS (~1.2 GB) en la primera ejecución:
```text
Loading base MMS model...
Model loaded on cpu (or cuda/mps).
Optimizing for CPU: limiting threads...
Precalentando modelo MMS en segundo plano...
✅ Modelo MMS precalentado en arranque del servidor.
[Dict] Loaded 1277 medical entries. Reverse entries: 840
 * Running on http://127.0.0.1:5000
```

### 3.4 Despliegue en Producción con Gunicorn y Nginx

#### Configuración de Gunicorn (`gunicorn_config.py`):
```python
import multiprocessing

bind = "0.0.0.0:5001"
workers = 1        # 1 worker para mantener estado en memoria de exportaciones y evitar sobrecarga de RAM por modelos IA
threads = 4        # Múltiples hilos para concurrencia HTTP
timeout = 300      # 300s para permitir inferencias largas en hardware modesto
wsgi_app = "app:app"
```

#### Ejecución del Servicio:
```bash
# Iniciar en segundo plano mediante tmux o systemd
gunicorn -c gunicorn_config.py
```

#### Configuración de Reverse Proxy Nginx (`/etc/nginx/sites-available/traductor`):
```nginx
server {
    listen 80;
    server_name tradkiche.zentary.net;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
```

```bash
# Habilitar sitio y configurar SSL con Certbot
sudo ln -sf /etc/nginx/sites-available/traductor /etc/nginx/sites-enabled/
sudo systemctl restart nginx
sudo certbot --nginx -d tradkiche.zentary.net
```

---

# 4. SEGURIDAD, BASE DE DATOS Y CONTROL DE ACCESO (RBAC)

### 4.1 Modelo Entidad-Relación

```mermaid
erDiagram
    USER ||--o{ AUDIT : "genera"
    USER {
        int id PK
        string username UK "Identificador único"
        string password_hash "PBKDF2 SHA256"
        string security_question "Pregunta secreta"
        string security_answer_hash "PBKDF2 SHA256"
        string role "admin | entrenador | basico"
        string status "pending | approved | rejected"
        string full_name "Nombre completo"
        string profile_image "Ruta de avatar"
        datetime last_login "Último acceso"
    }
    AUDIT {
        int id PK
        int user_id FK "user.id"
        string action "Descripción del evento"
        datetime timestamp "Fecha y hora UTC"
    }
```

### 4.2 Fragmento de Implementación SQLAlchemy y Decoradores RBAC

```python
# traductor_kiche/app.py
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    security_question = db.Column(db.String(200), nullable=False)
    security_answer_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='basico', nullable=False)
    status = db.Column(db.String(20), default='pending', nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    profile_image = db.Column(db.String(255), nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)

class Audit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

def role_required(*allowed_roles):
    """Protege rutas garantizando sesión activa y rol autorizado."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login_page'))
            user = User.query.get(session['user_id'])
            if not user or user.role not in allowed_roles:
                flash("No tienes permisos para acceder a esta sección.", "error")
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorator
    return decorator
```

### 4.3 Matriz de Permisos por Rol

| Módulo / Función | Rol: `basico` (Médico / Enfermero) | Rol: `entrenador` (Lingüista Nativo) | Rol: `admin` (Administrador IT) |
| :--- | :---: | :---: | :---: |
| Acceso al Traductor Clínico (`/`) | ✅ Permitido | ✅ Permitido | ✅ Permitido |
| Generación de Resumen de Triaje (`/generate-summary`) | ✅ Permitido | ✅ Permitido | ✅ Permitido |
| Gestión de "Mi Cuenta" (`/cuenta`) | ✅ Permitido | ✅ Permitido | ✅ Permitido |
| Modo Entrenamiento Guiado y Experto (`/train`) | ❌ Restringido | ✅ Permitido | ✅ Permitido |
| Recorte de Audio y Guardado de Dataset | ❌ Restringido | ✅ Permitido | ✅ Permitido |
| Práctica de Escritura (`/escritura`) | ❌ Restringido | ✅ Permitido | ✅ Permitido |
| Aprobación de Términos Médicos (`/revisar`) | ❌ Restringido | ✅ Permitido | ✅ Permitido |
| Panel de Administración de Usuarios (`/admin`) | ❌ Restringido | ❌ Restringido | ✅ Permitido |
| Gestión de Roles y Aprobación de Cuentas | ❌ Restringido | ❌ Restringido | ✅ Permitido |
| Configuración de API Keys y Reglas | ❌ Restringido | ❌ Restringido | ✅ Permitido |

---

# 5. GUÍA DE OPERACIÓN MÓDULO POR MÓDULO

---

## 5.1 MÓDULO 1: ACCESO, REGISTRO Y RECUPERACIÓN DE CUENTA

### 5.1.1 Diagrama del Proceso de Registro y Recuperación

```mermaid
sequenceDiagram
    autonumber
    actor Usuario
    participant Web as Interfaz Web
    participant Server as Servidor Flask
    participant DB as Base de Datos

    Note over Usuario,DB: Flujo A: Registro de Cuenta
    Usuario->>Web: Completa formulario con Pregunta Secreta
    Web->>Server: POST /register (Cifra password y respuesta con PBKDF2)
    Server->>DB: Guarda Usuario con status='pending'
    Server-->>Web: Redirige con aviso "Cuenta pendiente de aprobación"

    Note over Usuario,DB: Flujo B: Recuperación de Contraseña
    Usuario->>Web: Clic en "¿Olvidaste tu contraseña?"
    Usuario->>Web: Ingresa nombre de usuario
    Web->>Server: POST /recover (Paso 1: Validar usuario)
    Server->>DB: Consulta usuario
    Server-->>Web: Retorna la pregunta secreta configurada
    Usuario->>Web: Escribe la respuesta secreta
    Web->>Server: POST /recover (Paso 2: Comparar hash de respuesta)
    Server-->>Web: Respuesta correcta -> Habilita campo nueva contraseña
    Usuario->>Web: Ingresa y confirma nueva clave
    Web->>Server: POST /recover (Paso 3: Actualizar password_hash)
    Server->>DB: Actualiza hash de credencial
    Server-->>Web: Mensaje de confirmación -> Login
```

### 5.1.2 Pasos Operativos y Prompts de Imagen

#### Paso 1: Iniciar Sesión en Escritorio
* **Archivo asociado:** `01-login-escritorio.png`
* **Procedimiento:** Ingrese a la URL del sistema (`https://tradkiche.zentary.net` o `http://localhost:5000`). Digite su usuario y contraseña en la tarjeta central de vidrio esmerilado (*Liquid Glass*). Presione **Entrar**.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 01]:** Interfaz web de inicio de sesión moderna y elegante para el software médico Ch'aweb'al en pantalla de computadora de escritorio. Fondo fotográfico sutilmente desenfocado de un entorno clínico en Guatemala con telares tradicionales mayas de tonos verdes profundos (#102A25 y #059669). En el centro, una tarjeta flotante de estilo glassmorphism translúcido con borde iluminado suave. Encabezado con el logotipo oficial de Ch'aweb'al (símbolo maya estilizado y cruz médica verde esmeralda). Dos campos de formulario con etiquetas 'Usuario' y 'Contraseña' con tipografía sans-serif limpia (Inter). Botón principal verde esmeralda brillante con el texto 'Iniciar Sesión'. Enlace inferior dorado (#DB9D21) con el texto '¿Olvidaste tu contraseña?'. Esquina superior derecha con icono de usuario y selector de tema.

#### Paso 2 a 4: Menú de Cuenta y Solicitud de Registro
* **Archivos asociados:** `02-menu-cuentas.png`, `03-opcion-crear-cuenta.png`, `04-formulario-crear-cuenta.png`
* **Procedimiento:** Desde la pantalla de login, desplace el cursor al botón de usuario en la esquina superior derecha y seleccione **Crear cuenta**. Complete el formulario con Nombre Completo, Nombre de Usuario, Correo Institucional y Contraseña.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 04]:** Modal emergente de registro de usuario en interfaz web clínica. Título en verde esmeralda 'Solicitud de Cuenta Nueva - Ch'aweb'al'. Formulario de doble columna con campos: Nombre Completo, Nombre de Usuario único, Correo Electrónico institucional, Contraseña y Confirmar Contraseña con medidor visual de fortaleza de clave. Diseño limpio con tonos oscuros elegantes (#0F172A), bordes redondeados y botón inferior 'Enviar Solicitud de Registro'.

#### Paso 5 y 6: Pregunta de Seguridad y Estado de Aprobación
* **Archivos asociados:** `05-preguntas-seguridad.png`, `06-registro-pendiente-aprobacion.png`
* **Procedimiento:** Seleccione una de las preguntas de recuperación preestablecidas (ej. *"¿Cuál es el nombre de tu primera mascota?"*, *"¿En qué ciudad naciste?"*). Escriba su respuesta. Al guardar, el sistema mostrará un banner informativo indicando que la cuenta fue registrada y está a la espera de aprobación por parte del Administrador.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 06]:** Mensaje de alerta estilizado tipo toast flotante sobre el formulario de login. Fondo ámbar suave translúcido con icono de reloj de arena o escudo de seguridad. Texto principal: '¡Solicitud enviada con éxito! Tu cuenta está pendiente de activación por parte del administrador médico. Recibirás acceso en cuanto sea validada.' Botón verde para regresar al login.

#### Paso 7 a 12: Flujo de Recuperación de Tres Factores
* **Archivos asociados:** `07-validacion-correo.png`, `08-olvido-contrasena.png`, `09-recuperacion-usuario.png`, `10-usuario-no-encontrado.png`, `11-pregunta-seguridad.png`, `12-nueva-contrasena.png`
* **Procedimiento:**
  1. Haga clic en **¿Olvidaste tu contraseña?** en la pantalla de inicio de sesión.
  2. Ingrese su identificador de usuario y presione **Continuar**.
  3. El sistema buscará su registro. Si existe, presentará la pregunta de seguridad previamente configurada.
  4. Escriba su respuesta secreta. El backend verificará el hash criptográfico `pbkdf2:sha256`.
  5. Al superar la verificación, se desbloqueará el formulario para crear y confirmar la nueva contraseña.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 11]:** Pantalla de recuperación de acceso en Ch'aweb'al. Se muestra un cuadro central donde aparece el texto de la pregunta secreta: '¿Cuál es el segundo nombre de tu madre?'. Abajo un campo de texto activo con el cursor parpadeando para que el usuario responda, y un botón 'Validar Respuesta' con microanimación de brillo verde esmeralda.

#### Paso 13 a 18: Adaptabilidad Completa en Dispositivos Móviles y Modo PWA Standalone
* **Archivos asociados:** `13-login-movil.png` al `18-recuperacion-movil-completa.png`
* **Procedimiento:** En teléfonos móviles o tabletas clínicas de mano, el sistema apila los componentes verticalmente con áreas táctiles ampliadas (mínimo 48px por botón) para facilitar la operación médica con guantes.
* **Gesto de Recarga Rápida en WebApp Standalone (Pull-to-Refresh & Hold):** Al instalar la aplicación en el móvil (PWA sin barra de navegación del navegador), si el dispositivo pierde temporalmente conexión o la pantalla se congela, el usuario puede deslizar hacia abajo desde el tope superior de la pantalla (~120px) y sostener el dedo por un instante (450ms). Se despliega un indicador flotante superior de estilo glassmorphism que confirma con vibración háptica y recarga la página de forma limpia (`window.location.reload()`) con prevención de bloqueos de caché.

---

## 5.2 MÓDULO 2: TRADUCTOR CLÍNICO BIDIRECCIONAL

El núcleo del sistema permite comunicación médico-paciente en tiempo real con soporte bidireccional: **K'iche' → Español** y **Español → K'iche'**.

### 5.2.1 Diagrama del Pipeline de Inferencia de Voz (ASR & TTS)

```mermaid
graph TD
    A[Voz del Paciente en K'iche'] -->|WebRTC / MediaRecorder| B[Audio WebM / WAV 16kHz]
    B --> C[Eliminación de Offset DC: audio - mean]
    C --> D[Normalización de Pico: audio / max_abs]
    D --> E[Recorte de Silencios: Librosa trim top_db=30]
    E --> F[Inferencia Meta MMS-1B Wav2Vec2 + LoRA PEFT]
    F --> G[Transcripción Cruda en Caracteres Fonéticos]
    G --> H[Conversor Ortográfico ALMG & Reglas Fonéticas]
    H --> I{Búsqueda en Diccionario Médico}
    I -->|Coincidencia Exacta o Glotal| J[Traducción Médica Oficial]
    I -->|Coincidencia Difusa RapidFuzz >= 88| J
    I -->|Sin Coincidencia en Diccionario| K[Traducción Neuronal Fallback NLLB-200]
    J --> L[Activación Visual de Modelo Anatómico]
    K --> L
    L --> M[Síntesis gTTS en Español para el Personal Médico]
    L --> N[Acumulación en Contador de Sesión Clínica]
```

### 5.2.2 Fragmento de Código del Motor de Inferencia (`/translate`)

```python
# traductor_kiche/app.py
@app.route('/translate', methods=['POST'])
def translate():
    global model, processor, device
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    filename = f"{uuid.uuid4()}.webm"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    try:
        # 1. Carga uniforme a 16kHz Mono Float32
        audio_input = load_audio_file(filepath, target_sr=16000)

        # 2. Recorte de silencios para evitar alucinaciones acústicas
        audio_input, _ = librosa.effects.trim(audio_input, top_db=30)

        # 3. Filtrado de offset DC y normalización de volumen
        audio_input = audio_input - np.mean(audio_input)
        if np.max(np.abs(audio_input)) > 0:
            audio_input = audio_input / np.max(np.abs(audio_input))

        # 4. Inferencia con Wav2Vec2
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        ids = torch.argmax(outputs.logits, dim=-1)[0]
        transcription = processor.decode(ids)

        # 5. Normalización según Academia de Lenguas Mayas (ALMG)
        display_transcription = to_almg_display(transcription)
        display_transcription = apply_normalization_rules(display_transcription)

        # 6. Traducción clínica ponderada
        translation = translate_kiche_to_spanish(display_transcription)

        return jsonify({
            'transcription': display_transcription,
            'translation': translation,
            'audio_url': None
        })
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
```

### 5.2.3 Pasos Operativos y Prompts de Imagen

#### Paso 19: Traductor K'iche' a Español (Pantalla Principal)
* **Archivo asociado:** `19-traductor-inicial.png`
* **Procedimiento:** Al ingresar con una cuenta autorizada, se presenta la pantalla principal con el micrófono en reposo (verde jade). En computadoras puede presionar la tecla `Ctrl` (o `Option` en Mac) o hacer clic sostenido en el micrófono para dictar.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 19]:** Pantalla principal de trabajo del Traductor Clínico Ch'aweb'al. Interfaz oscura ultra-profesional (#0F172A). En el centro, un botón circular gigante para el micrófono rodeado de un sutil anillo resplandeciente en verde jade (#059669). Debajo, un texto informativo: 'Mantén presionado para hablar en K'iche' (o presiona Ctrl)'. En la parte superior, un interruptor deslizable elegante que marca 'K'iche' → Español' como activo, acompañado de botones redondeados de navegación a Cuenta, Modo Lite y Centro de Ayuda.

#### Paso 20: Captura de Voz y Estado de Escucha
* **Archivo asociado:** `20-microfono-escuchando.png`
* **Procedimiento:** Al mantener presionado el botón o la tecla de atajo, el botón del micrófono se transforma inmediatamente a color rojo carmesí (`#C53520`) con una animación de ondas expansivas de audio (*audio wave ripple*), indicando que el hardware de captura está digitalizando a 16 kHz.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 20]:** Primer plano del botón de micrófono en estado activo de grabación. El botón circular cambia a rojo brillante (#C53520) con un icono de micrófono en color blanco y ondas concéntricas animadas que se expanden hacia afuera simulando la recepción de ondas sonoras. Debajo, un indicador de texto dinámico: 'Escuchando paciente en K'iche'...'.

#### Paso 21: Presentación de la Traducción y Modelo Anatómico
* **Archivo asociado:** `21-traduccion-y-modelo-anatomico.png`
* **Procedimiento:** Al soltar el micrófono, el sistema procesa el audio. En pantalla aparece la transcripción en K'iche' estándar (editable por si hubo ruido ambiental), el texto traducido al español en tipografía grande y legible, y a la par se despliega la **Asistencia Gráfica Anatómica**, iluminando en rojo o ámbar la zona corporal mencionada (cabeza, pecho, estómago, brazos o piernas).
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 21]:** Cuadro de resultados del traductor clínico. A la izquierda: texto superior en verde esmeralda con la frase en K'iche' ('K'ax nu jolom'), línea divisoria sutil, y texto principal grande en blanco 'Me duele la cabeza' con botón para reproducir audio en español. A la derecha, silueta vectorial minimalista del cuerpo humano donde la zona de la cabeza se ilumina en rojo cálido pulsante, indicando la región anatómica con dolor reportado por el paciente.

#### Paso 22 a 25: Contador de Consultas y Resumen Clínico Estructurado
* **Archivos asociados:** `22-contador-y-resumen.png`, `23-resumen-clinico-inicial.png`, `24-resumen-clinico-generado.png`, `25-copiar-resumen-clinico.png`
* **Procedimiento:**
  1. En la esquina inferior derecha aparece un botón flotante con el icono de expediente médico (`clinical_notes`) y una burbuja roja con el conteo de síntomas traducidos durante la consulta.
  2. Al pulsar el botón flotante, se abre el modal **Resumen Clínico**.
  3. Haga clic en **Generar Ahora**. La función `format_consultation_summary` agrupa y desduplica cronológicamente las intervenciones del personal de salud y las respuestas del paciente traducidas exclusivamente al español.
  4. Presione el botón **Copiar** (`content_paste`) para transferir el informe al portapapeles y pegarlo directamente en el software de expediente electrónico o historial clínico del hospital.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 24]:** Modal de 'Resumen Clínico de la Consulta' en estilo glassmorphism. Contiene un área de texto grande con tipografía monospace limpia mostrando un informe médico formateado: 'RESUMEN CLÍNICO DE LA CONSULTA (ESPAÑOL) / 1. Intervenciones del personal: "¿Dónde siente el malestar?" / 2. Síntomas reportados por el paciente: Dolor intenso en el estómago, mareo de dos días'. En la parte inferior, botón verde 'Copiar al portapapeles' con icono de confirmación y botón 'Cerrar'.

#### Paso 26 a 28: Traducción Inversa (Español a K'iche')
* **Archivos asociados:** `26-cambio-espanol-kiche.png`, `27-espanol-kiche-escucha.png`, `28-selector-idioma.png`
* **Procedimiento:** Haga clic en el conmutador superior para activar **Español → K'iche'**. El tema se ajusta con acentos en color ámbar/naranja. El médico habla en español; el sistema transcribe en español, busca la correspondencia en K'iche' y reproduce automáticamente el audio nativo o sintetizado en K'iche' para que el paciente escuche las indicaciones en su lengua materna.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 26]:** Interfaz del traductor en modo inverso 'Español → K'iche''. El interruptor superior resalta la dirección seleccionada con un fondo ámbar cálido (#DB9D21). El botón del micrófono muestra un resplandor naranja suave. Texto de ayuda: 'Mantén presionado para hablar en Español y traducir a K'iche''.

#### Paso 29: Tour Virtual Interactivo
* **Archivo asociado:** `29-ayuda-tour-virtual.png`
* **Procedimiento:** Al presionar el icono de ayuda (`help`) ubicado al lado del selector de idiomas, se activa una guía interactiva que sombrea la pantalla y coloca reflectores (*tooltips*) explicativos paso a paso sobre los controles del traductor.

---

## 5.3 MÓDULO 3: PREFERENCIAS, RENDIMIENTO Y CUENTA

### 5.3.1 Pasos Operativos y Prompts de Imagen

#### Paso 30 y 31: Modo Lite para Equipos de Bajo Rendimiento
* **Archivos asociados:** `30-acceso-modo-lite.png`, `31-confirmacion-modo-lite.png`
* **Procedimiento:** En puestos de salud rurales con computadoras de bajos recursos, presione el icono de velocímetro (`speed`) en la barra superior. El **Modo Lite** desactiva los filtros de desenfoque (*backdrop-filter: blur*), sombras gaussianas pesadas y animaciones complejas de entrada, reduciendo el consumo de CPU y memoria RAM en más de un 60%.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 30]:** Barra de herramientas superior de Ch'aweb'al con el cursor posicionado sobre el icono de velocímetro amarillo ('Modo Lite'). Se despliega un tooltip oscuro: 'Modo Lite: optimiza el rendimiento desactivando animaciones y desenfoques para equipos modestos'. Al hacer clic, aparece un aviso toast verde: 'Modo Lite Activado'.

#### Paso 32 a 34: Gestión del Perfil ("Mi Cuenta")
* **Archivos asociados:** `32-acceso-mi-cuenta.png`, `33-mi-cuenta.png`, `34-icono-mi-cuenta.png`
* **Procedimiento:** Haga clic en el botón de usuario (`person`) en la barra superior para ingresar a `/cuenta`. Aquí puede actualizar su Nombre Completo, visualizar su Rol y Estado en el sistema, y actualizar su contraseña ingresando la clave actual y la nueva contraseña confirmada.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 33]:** Pantalla de perfil de usuario 'Mi Cuenta'. Tarjeta central dividida en dos secciones: a la izquierda, avatar de usuario con badge verde que indica 'Rol: Médico General / Estado: Aprobado'. A la derecha, formulario para cambiar nombre completo y bloque para actualización segura de contraseña (contraseña actual, nueva contraseña y confirmar contraseña) con botón 'Guardar Cambios'.

---

## 5.4 MÓDULO 4: ENTRENAMIENTO DE VOZ Y EXPANSIÓN DE DATASETS

El módulo `/train` está reservado para usuarios con rol `entrenador` y `admin`. Su objetivo es recopilar muestras de audio validadas por hablantes nativos para reentrenar el modelo ASR MMS y alimentar el corpus lingüístico.

### 5.4.1 Diagrama del Pipeline de Entrenamiento

```mermaid
graph LR
    subgraph ModoGuiado ["1. Modo Guiado"]
        A1[Selección de Categoría Médica] --> A2[Frase Curada en K'iche']
        A2 --> A3[Grabación de Voz de Hablante Nativo]
        A3 --> A4[Inferencia MMS CTC de Control]
        A4 --> A5[Guardado en CSV y Almacenamiento Local]
    end

    subgraph ModoExperto ["2. Modo Experto"]
        B1[Ingreso de Frase Libre K'iche' & Español] --> B2[Captura o Subida de Audio WAV/MP3]
        B2 --> B3[Editor Visual de Forma de Onda Waveform]
        B3 --> B4[Recorte Interactivo de Silencios y Ruido]
        B4 --> B5[Test de Inferencia y Confirmación]
    end

    A5 --> S3Sync[Sincronización con AWS S3 / Exportación ZIP]
    B5 --> S3Sync
```

### 5.4.2 Fragmento de Código del Editor de Recorte de Audio

```python
# Segmento de app.py para procesamiento y guardado de muestras de entrenamiento
@app.route('/save-training-sample', methods=['POST'])
@role_required('admin', 'entrenador')
def save_training_sample():
    kiche_text = request.form.get('kiche_text', '').strip()
    spanish_text = request.form.get('spanish_text', '').strip()
    audio_file = request.files.get('audio')

    if not audio_file or not kiche_text:
        return jsonify({'error': 'Faltan datos obligatorios'}), 400

    filename = f"train_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    save_path = os.path.join(TRAINING_AUDIO_DIR, filename)

    # Convertir y guardar a WAV PCM 16kHz Mono
    audio_data = load_audio_file(audio_file, target_sr=16000)
    sf.write(save_path, audio_data, 16000, subtype='PCM_16')

    # Registrar en el dataset CSV
    with open(TRAINING_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([filename, kiche_text, spanish_text, session.get('username', 'anonimo')])

    # Sincronizar en segundo plano con S3 si está configurado
    sync_file_to_s3(save_path, f"training_audio/{filename}")

    return jsonify({'success': True, 'filename': filename})
```

### 5.4.3 Pasos Operativos y Prompts de Imagen

#### Paso 35 a 40: Modo Guiado para Hablantes Nativos
* **Archivos asociados:** `35-modo-entrenamiento.png` al `40-validar-transcripcion-guiada.png`
* **Procedimiento:**
  1. Ingrese a la ruta `/train` o presione el icono de birrete universitario (`school`).
  2. Seleccione una categoría médica (ej. *Partes del cuerpo, Síntomas digestivos, Fiebre y dolor*).
  3. El sistema muestra la frase oficial en K'iche' con su traducción de referencia.
  4. Presione el botón de grabación y pronuncie la frase con naturalidad.
  5. El sistema ejecuta el reconocimiento MMS y compara lo dicho con el texto esperado. Si la transcripción es correcta, presione **Guardar Aporte**.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 36]:** Pantalla de Modo Guiado en el módulo de entrenamiento. Tarjeta central con una tarjeta destacada con borde púrpura suave. Frase en K'iche' grande en negrita: 'K'ax nu ch'u'k' acompañada de su traducción 'Me duele el codo'. Debajo, selector de categorías con chips redondeados ('Anatomía', 'Dolor', 'Urgencias'). Botón central de grabación con micrófono violeta y botones laterales para 'Frase Siguiente' y 'Frase Aleatoria'.

#### Paso 41 a 48: Modo Experto y Editor de Recorte de Audio (Waveform Trimmer)
* **Archivos asociados:** `41-selector-modo-guiado-experto.png` al `48-confirmacion-dataset.png`
* **Procedimiento:**
  1. Conmute al **Modo Experto**.
  2. Escriba libremente la frase médica en español y su traducción precisa en K'iche'.
  3. Grabe con su micrófono o suba un archivo de audio existente (`.wav`, `.mp3`, `.ogg`, `.aiff`).
  4. Se desplegará el **Editor de Forma de Onda (Waveform)** con marcadores deslizables izquierdo y derecho para delimitar con exactitud el inicio y fin de la voz, eliminando chasquidos de teclado o ruidos de fondo.
  5. Pulse **Recortar y enviar a MMS** para escuchar el fragmento y verificar la salida del modelo.
  6. Presione **Guardar frase y audio** para incorporarlo formalmente al corpus de entrenamiento.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 44]:** Visualizador interactivo de forma de onda (Waveform Audio Trimmer) en el modo experto de entrenamiento. Se muestra un gráfico de ondas sonoras de audio color turquesa sobre fondo oscuro. Dos barras delimitadoras verticales semitransparentes permiten ajustar los puntos de corte inicial y final. Botones de acción inferiores: 'Reproducir selección', 'Recortar y probar con MMS' y 'Confirmar guardado en dataset'.

---

## 5.5 MÓDULO 5: PRÁCTICA DE ESCRITURA Y REVISIÓN LINGÜÍSTICA

Ubicado en `/escritura` y `/revisar`, permite a los lingüistas validar la ortografía estándar según las pautas de la ALMG y enriquecer el diccionario médico.

### 5.5.1 Pasos Operativos y Prompts de Imagen

#### Paso 49 y 50: Práctica de Escritura y Asistente IA
* **Archivos asociados:** `49-practica-escritura.png`, `50-generar-frase-o-personalizada.png`
* **Procedimiento:** Ingrese a `/escritura`. El usuario puede practicar transcribiendo frases dictadas o solicitar a la IA de Google Gemini que genere nuevas combinaciones de oraciones médicas basadas en la sintaxis K'iche'. Si propone una frase personalizada no registrada, queda en cola de espera.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 50]:** Pantalla de práctica de escritura K'iche'. Área de redacción con teclado virtual o atajos para insertar caracteres especiales y glotales (', b', ch', k', q', t', tz'). Botón superior con icono de chispa brillante 'Generar frase médica con IA (Gemini)' y campo para ingresar propuestas de nuevas variantes lingüísticas.

#### Paso 51 y 52: Panel de Revisión y Clasificación Médica
* **Archivos asociados:** `51-acceso-panel-revision.png`, `52-panel-revision.png`
* **Procedimiento:** Los administradores y lingüistas acceden a `/revisar` para examinar todas las frases propuestas. Pueden corregir apóstrofes o erratas, asignar la categoría médica correspondiente y hacer clic en **Aprobar**. Las frases aprobadas se integran inmediatamente a la sección `_escritura_aprobada` del diccionario médico en caliente.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 52]:** Panel administrativo de revisión lingüística. Tabla con lista de frases pendientes de moderación: columnas para Frase en Español, Frase en K'iche', Usuario proponente y Selector de Categoría Médica. En la columna de acciones, botón verde con icono de visto bueno 'Aprobar para Diccionario' y botón rojo con icono de basurero 'Descartar'.

---

## 5.6 MÓDULO 6: PANEL DE ADMINISTRACIÓN Y AUDITORÍA

Exclusivo para usuarios con rol `admin` en la ruta `/admin`.

### 5.6.1 Pasos Operativos y Prompts de Imagen

#### Paso 53 a 57: Aprobación y Gestión Integral de Usuarios
* **Archivos asociados:** `53-panel-administrador-pendientes-anonimo.png` al `57-crear-nuevo-usuario.png`
* **Procedimiento:**
  1. El panel muestra en la parte superior las solicitudes de usuarios con estado `pending` resaltadas con insignia amarilla.
  2. El administrador puede verificar el nombre y correo del solicitante y hacer clic en **Aprobar Cuenta**.
  3. Para usuarios existentes, puede cambiar el rol en cualquier momento seleccionando entre:
     - `basico`: Personal de salud general (solo consulta clínica).
     - `entrenador`: Lingüistas y recolectores de dialectos.
     - `admin`: Control absoluto del sistema.
  4. Cuenta con la opción **Crear Usuario Directo** para registrar personal médico sin que deban pasar por el flujo de registro público.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 53]:** Panel de control de administración de Ch'aweb'al en modo oscuro. Arriba, tarjetas de métricas: 'Usuarios Totales: 14', 'Solicitudes Pendientes: 3', 'Frases en Dataset: 1,280'. Abajo, tabla de gestión con filas alternadas; las cuentas pendientes tienen un badge amarillo brillante 'Pendiente'. Botones de acción desplegable para aprobar, editar rol o revocar acceso.

#### Paso 58: Conmutador de Apariencia (Modo Claro / Modo Oscuro)
* **Archivo asociado:** `58-modo-claro.png`
* **Procedimiento:** Ubicado en la esquina superior de todas las pantallas, el botón con icono de sol/luna alterna entre el tema oscuro por defecto y el tema claro (fondo marfil clínico `#F7F1E7` con texto en verde bosque profundo `#102A25`), manteniendo siempre óptimo contraste visual conforme a los estándares de accesibilidad WCAG AAA.
* **Prompt para Generar / Renderizar Imagen:**
> **[PROMPT VISUAL - IMAGEN 58]:** Interfaz de Ch'aweb'al en Modo Claro. Fondo general marfil suave (#F7F1E7), tarjetas con sombras tenues y bordes limpios, textos principales en verde oscuro elegante (#102A25) y botones de acción en verde esmeralda (#10B981). Se observa el botón de alternancia en la esquina superior derecha mostrando el icono de sol activo.

#### 5.6.2 Zona Protegida: Gestor y Carga en Caliente de Modelos de IA (LoRA Colab)
* **Propósito:** Permite al administrador cargar directamente desde la web nuevos modelos entrenados en Google Colab (`mms_kiche_trained.zip`) y activarlos en caliente sin necesidad de acceder por SSH a la terminal ni reiniciar el servidor.
* **Acceso Resguardado:** Por motivos de seguridad operativa, esta función está resguardada tras una puerta de autenticación (`aiModelAuthModal`) que exige reconfirmar la contraseña del administrador activo.
* **Flujo Operativo:**
  1. El administrador hace clic en el botón morado **Motor de IA (LoRA)** en el panel de administración.
  2. Ingresa su contraseña de administrador para desbloquear el **AI Model Studio**.
  3. Puede visualizar el estado en tiempo real del modelo en memoria: modo activo (`lora` o `generic`), tamaño de los pesos, fecha y hardware activo (`GPU`, `MPS` o `CPU`).
  4. Cuenta con un botón **Probar Inferencia en Vivo** para verificar la latencia y operatividad de PyTorch en milisegundos.
  5. **Carga de Nuevo Modelo:** Arrastra el archivo `mms_kiche_trained.zip` descargado de Colab. El sistema:
     - Valida la presencia de `adapter_config.json` y `adapter_model.safetensors`.
     - Crea una copia de seguridad automática del modelo anterior en `models/backups/`.
     - Extrae los nuevos pesos en `models/mms_kiche_trained/`.
     - Sincroniza automáticamente los nuevos archivos con AWS S3 si está configurado.
     - Ejecuta una recarga en caliente (*Hot-Reload*) bajo `model_lock` y precalienta el grafo de tensores.
  6. **Conmutador Rápido:** Permite alternar con un solo clic entre el modelo fino entrenado en Colab y el modelo genérico oficial de Meta MMS.
  7. **Rollback de Respaldos:** Permite restaurar cualquier punto de control anterior con un solo clic si el nuevo entrenamiento no cumple con las expectativas.

---

# 6. REFERENCIA DE LA API REST

Todos los endpoints que reciben audio o devuelven información estructurada se documentan a continuación para fines de integración con sistemas externos (expedientes electrónicos de salud, aplicaciones móviles Android/iOS):

### 6.1 `POST /translate` (Audio K'iche' → Traducción Español)
* **Descripción:** Procesa un archivo de audio con voz en K'iche' y retorna la transcripción ALMG y la traducción médica al español.
* **Headers:** `Content-Type: multipart/form-data`
* **Parámetros Form:**
  - `audio`: Archivo binario de audio (`.webm`, `.wav`, `.ogg`).
  - `is_warmup`: `true` | `false` (Opcional, para precalentamiento del modelo).
* **Respuesta Exitosa (`200 OK`):**
```json
{
  "transcription": "k'ax nu jolom",
  "translation": "me duele la cabeza",
  "audio_url": null
}
```

### 6.2 `POST /translate-es-audio` (Audio Español → Traducción K'iche')
* **Descripción:** Procesa la voz en español del médico, transcribe mediante Whisper y retorna el equivalente en K'iche' junto con el enlace al audio nativo si existe.
* **Headers:** `Content-Type: multipart/form-data`
* **Parámetros Form:**
  - `audio`: Archivo binario de voz en español.
* **Respuesta Exitosa (`200 OK`):**
```json
{
  "transcription": "¿dónde le duele?",
  "translation": "¿jawichi' k'ax wi?",
  "audio_url": "/training-audio/train_indicacion_donde_duele.wav"
}
```

### 6.3 `POST /translate-to-kiche` (Texto Español → K'iche')
* **Descripción:** Traduce una cadena de texto en español hacia idioma K'iche'.
* **Headers:** `Content-Type: application/json`
* **Body:**
```json
{
  "text": "tome una tableta de medicina cada ocho horas"
}
```
* **Respuesta Exitosa (`200 OK`):**
```json
{
  "original": "tome una tableta de medicina cada ocho horas",
  "translation": "tij jun kunb'al pa wajxaqib' ramaj",
  "audio_url": "/training-audio/indicacion_tableta_8h.wav"
}
```

### 6.4 `POST /generate-summary` (Generación de Resumen de Triaje)
* **Descripción:** Recibe el arreglo cronológico de frases dictadas en la consulta y devuelve el texto estructurado del expediente.
* **Headers:** `Content-Type: application/json`
* **Body:**
```json
{
  "items": [
    {
      "direction": "espanol_a_kiche",
      "espanol": "¿Cuál es su malestar?",
      "kiche": "¿Jas ri a yabil?"
    },
    {
      "direction": "kiche_a_espanol",
      "espanol": "Tengo ardor intenso en el estómago",
      "kiche": "K'aq'an nu pam"
    }
  ]
}
```
* **Respuesta Exitosa (`200 OK`):**
```json
{
  "summary": "RESUMEN CLÍNICO DE LA CONSULTA (ESPAÑOL)\n────────────────────────────────────────\n1. Intervenciones y preguntas formuladas (Español):\n   • \"¿Cuál es su malestar?\"\n\n2. Síntomas y respuestas del paciente (Español):\n   • \"Tengo ardor intenso en el estómago\"\n"
}
```

### 6.5 `POST /api/start-export` y `GET /download-export/<task_id>`
* **Descripción:** Inicia la compresión ZIP de todas las grabaciones de voz y el dataset CSV en segundo plano con control de progreso para descarga masiva de entrenamiento.

---

# 7. MOTOR LINGÜÍSTICO Y DICCIONARIO MÉDICO

### 7.1 Jerarquía Ponderada de Búsqueda (5 Niveles)

Cuando el sistema recibe una transcripción o texto para traducir, ejecuta la búsqueda a través de 5 niveles de precedencia para garantizar precisión técnica:

```mermaid
graph TD
    In[Texto a Traducir] --> N1[1. escritura_dataset.csv - Máxima Prioridad / Validado por Expertos]
    N1 -->|No encontrado| N2[2. _escritura_aprobada en JSON - Frases Aprobadas en Panel]
    N2 -->|No encontrado| N3[3. Secciones Médicas Principales de medical_dictionary.json]
    N3 -->|No encontrado| N4[4. TRAINING_PHRASES - Frases Semilla de Entrenamiento]
    N4 -->|No encontrado| N5[5. ia_generado en JSON - Propuestas de IA / Mínima Prioridad]
    N5 -->|No encontrado| N6[6. Búsqueda Difusa RapidFuzz Ratio >= 88%]
    N6 -->|No encontrado| N7[7. Fallback Neuronal NLLB / Deep-Translator]
```

### 7.2 Normalización de Apóstrofes y Reglas Fonéticas

El idioma K'iche' utiliza consonantes glotalizadas que los teclados convencionales o motores de transcripción pueden capturar con diferentes símbolos tipográficos (`'`, `’`, `ꞌ`, `` ` ``). El motor normaliza todas las variantes al apóstrofe estándar reconocido por la ALMG:

```python
def normalize_kiche_for_cache(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = text.replace("ꞌ", "'").replace("’", "'").replace("`", "'")
    text = re.sub(r'[¿?¡!.,;:\"()\[\]{}]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def strip_glottal(text):
    """Permite coincidencia fonética incluso si el usuario omitió la glotal."""
    return re.sub(r"[']", "", normalize_kiche_for_cache(text))
```

Además, el archivo `normalization_rules.json` almacena correcciones dialectales directas configurables en caliente por el administrador:
```json
{
    "nukab": "nuq'ab'",
    "qax": "k'ax",
    "nujolom": "nu jolom"
}
```

---

# 8. RESOLUCIÓN DE PROBLEMAS (TROUBLESHOOTING) Y MANTENIMIENTO

### 8.1 Problemas Frecuentes y Acciones Correctivas

#### A. La primera traducción tarda varios segundos en responder
* **Causa:** El modelo Meta MMS-1B Wav2Vec2 y los adaptadores LoRA son cargados y precalentados en memoria al recibir la primera solicitud si el servidor se acaba de reiniciar.
* **Solución:** Comportamiento esperado en la versión 2.1. El frontend ejecuta automáticamente un paquete silencioso de precalentamiento (`is_warmup=true`) en cuanto el usuario carga la pantalla del traductor para que el retardo ocurra antes de que el paciente empiece a hablar.

#### B. Error de permisos de micrófono en el navegador (`NotAllowedError`)
* **Causa:** La API `navigator.mediaDevices.getUserMedia` exige una conexión segura vía HTTPS.
* **Solución:** Asegúrese de ingresar con dominio seguro `https://` o `http://localhost`. En navegadores Chrome, verifique en el candado de la barra de direcciones que el permiso de *Micrófono* esté configurado en "Permitir".

#### C. El servidor se cierra repentinamente en AWS EC2 (Proceso Terminado / Killed)
* **Causa:** Agotamiento de la memoria física RAM durante la inferencia en instancias `t2.micro` (1 GB RAM).
* **Solución:** Ejecute inmediatamente el script de configuración de memoria Swap:
  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```

#### D. Las frases aprobadas no se reflejan en el traductor
* **Causa:** El caché del diccionario en memoria requiere actualización.
* **Solución:** Ingrese a la ruta `/reload-dict` con credenciales de administrador o reinicie el servicio Gunicorn. El sistema recargará los mapas hash en menos de 200 ms.

### 8.2 Protocolo de Respaldo y Sincronización con AWS S3

Para realizar un respaldo completo de la base de datos de usuarios y los archivos de audio de entrenamiento hacia el bucket seguro de Amazon S3, ejecute:

```bash
# Exportación manual de seguridad
./export_training_data.sh
```

El script genera un archivo compactado `kiche_dataset_backup_YYYYMMDD.zip` conteniendo:
1. `app.db` (Usuarios, permisos y tablas de auditoría).
2. `medical_dictionary.json` y `normalization_rules.json`.
3. `escritura_dataset.csv` y registros de entrenamiento.
4. Directorio completo `uploads/` y `training_data/`.

---

> **Aviso de Confidencialidad y Ética Médica:** El sistema Ch'aweb'al no almacena grabaciones de voz ni datos personales identificables de las consultas médicas realizadas en el módulo traductor. Las traducciones residen de forma volátil en la memoria de la sesión activa del navegador únicamente mientras se formula el resumen clínico. Al finalizar o recargar la página, las memorias temporales se eliminan, preservando en todo momento el secreto profesional médico y la privacidad del paciente.
