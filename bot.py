import discord
from discord import app_commands, Embed, Color, File
from discord.ext import commands, tasks
import os
import io
import json
import base64
import asyncio
import datetime
import random
import re
import time
import logging
import traceback
import aiohttp
import google.generativeai as genai
import requests
from collections import defaultdict, deque
from typing import Dict, List, Optional, Union, Tuple, Set
from dotenv import load_dotenv
import sqlite3
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO

# ================ CONFIGURACIÓN DE LOGGING ================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Eliminado el FileHandler para evitar escribir en el sistema de archivos efímero
    ]
)
logger = logging.getLogger("DiscordBot")

# ================ CARGA DE VARIABLES DE ENTORNO ================
# Primero intentamos cargar desde .env para desarrollo local
load_dotenv()

# Tokens y claves API
TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY') or None

# ================ CONFIGURACIÓN DE LA BASE DE DATOS ================
class Database:
    def __init__(self, db_path="bot_data.db"):
        self.db_path = db_path
        self.connection = None
        self.initialize_db()
    
    def get_connection(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        return self.connection
    
    def initialize_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Tabla para configuración de canales
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS channel_config (
            guild_id INTEGER,
            channel_id INTEGER PRIMARY KEY,
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabla para perfiles de usuario
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            first_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            interaction_count INTEGER DEFAULT 0,
            last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            custom_greeting TEXT,
            preferences TEXT
        )
        ''')
        
        # Tabla para historial de conversaciones
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            channel_id INTEGER,
            message TEXT,
            response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_profiles (user_id),
            FOREIGN KEY (channel_id) REFERENCES channel_config (channel_id)
        )
        ''')
        
        # Tabla para estadísticas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            total_messages INTEGER DEFAULT 0,
            total_images INTEGER DEFAULT 0,
            active_users INTEGER DEFAULT 0
        )
        ''')
        
        conn.commit()
    
    def set_active_channel(self, guild_id, channel_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Primero desactivar todos los canales para este servidor
        cursor.execute(
            "UPDATE channel_config SET active = 0 WHERE guild_id = ?",
            (guild_id,)
        )
        
        # Luego configurar el nuevo canal activo
        cursor.execute(
            '''
            INSERT OR REPLACE INTO channel_config (guild_id, channel_id, active, updated_at)
            VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            ''',
            (guild_id, channel_id)
        )
        
        conn.commit()
        return True
    
    def get_active_channels(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT guild_id, channel_id FROM channel_config WHERE active = 1")
        return cursor.fetchall()
    
    def is_channel_active(self, channel_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT active FROM channel_config WHERE channel_id = ?", (channel_id,))
        result = cursor.fetchone()
        return result and result["active"] == 1
    
    def update_user_interaction(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Insertar o actualizar el perfil del usuario
        cursor.execute(
            '''
            INSERT INTO user_profiles (user_id, interaction_count, last_interaction)
            VALUES (?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                interaction_count = interaction_count + 1,
                last_interaction = CURRENT_TIMESTAMP
            ''',
            (user_id,)
        )
        
        conn.commit()
    
    def get_user_profile(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        return cursor.fetchone()
    
    def set_custom_greeting(self, user_id, greeting):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO user_profiles (user_id, custom_greeting)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                custom_greeting = ?
            ''',
            (user_id, greeting, greeting)
        )
        
        conn.commit()
    
    def save_conversation(self, user_id, channel_id, message, response):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO conversation_history 
            (user_id, channel_id, message, response)
            VALUES (?, ?, ?, ?)
            ''',
            (user_id, channel_id, message, response)
        )
        
        conn.commit()
    
    def get_recent_conversations(self, user_id, limit=5):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            SELECT message, response, timestamp
            FROM conversation_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''',
            (user_id, limit)
        )
        
        return cursor.fetchall()
    
    def update_daily_stats(self, messages=0, images=0, user_id=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Actualizar o crear estadísticas para hoy
        cursor.execute(
            '''
            INSERT INTO stats (date, total_messages, total_images, active_users)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(date) DO UPDATE SET
                total_messages = total_messages + ?,
                total_images = total_images + ?
            ''',
            (today, messages, images, messages, images)
        )
        
        # Si se proporciona un user_id, contar como usuario activo único
        if user_id:
            cursor.execute(
                '''
                WITH active_users AS (
                    SELECT COUNT(DISTINCT user_id) as count
                    FROM conversation_history
                    WHERE date(timestamp) = ?
                )
                UPDATE stats
                SET active_users = (SELECT count FROM active_users)
                WHERE date = ?
                ''',
                (today, today)
            )
        
        conn.commit()
    
    def get_stats(self, days=7):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            SELECT date, total_messages, total_images, active_users
            FROM stats
            ORDER BY date DESC
            LIMIT ?
            ''',
            (days,)
        )
        
        return cursor.fetchall()
    
    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

# ================ MEMORIA DEL BOT ================
class BotMemory:
    def __init__(self, max_memory_per_user=10):
        self.max_memory_per_user = max_memory_per_user
        self.user_memories = defaultdict(lambda: deque(maxlen=max_memory_per_user))
        self.global_context = {}
    
    def add_interaction(self, user_id, message, response):
        """Añade una interacción a la memoria de un usuario"""
        self.user_memories[user_id].append({
            "message": message,
            "response": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def get_recent_interactions(self, user_id, limit=None):
        """Obtiene las interacciones recientes de un usuario"""
        if limit is None:
            return list(self.user_memories[user_id])
        return list(self.user_memories[user_id])[-limit:]
    
    def clear_user_memory(self, user_id):
        """Limpia la memoria de un usuario específico"""
        self.user_memories[user_id].clear()
    
    def set_global_context(self, key, value):
        """Establece un valor en el contexto global"""
        self.global_context[key] = value
    
    def get_global_context(self, key, default=None):
        """Obtiene un valor del contexto global"""
        return self.global_context.get(key, default)

# ================ CONFIGURACIÓN DE IA ================
class AIHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.available_models = []
        self.current_model = None
        self.initialize_model()
    
    def initialize_model(self):
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.0-pro',
            'gemini-pro',
            'gemini-pro-latest',
            'gemini-1.5-pro'
        ]
        
        for name in model_names:
            try:
                logger.info(f"Intentando inicializar modelo: {name}")
                model = genai.GenerativeModel(name)
                # Hacer una prueba simple
                response = model.generate_content("Test")
                logger.info(f"Modelo {name} inicializado correctamente")
                self.available_models.append(name)
                self.current_model = model
                self.model_name = name
                break
            except Exception as e:
                logger.error(f"Error con modelo {name}: {e}")
        
        if not self.current_model:
            logger.critical("No se pudo inicializar ningún modelo de Gemini")
    
    async def generate_response(self, prompt, system_prompt=None):
        if not self.current_model:
            return "No se pudo inicializar el modelo de IA. Por favor, verifica la configuración."
        
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUsuario: {prompt}"
            else:
                full_prompt = prompt
                
            response = self.current_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error al generar respuesta: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
    
    async def translate_text(self, text, target_language="English"):
        if not self.current_model:
            return text
        
        try:
            prompt = f"Translate the following text to {target_language}. Return ONLY the translation, no explanations:\n\n{text}"
            response = self.current_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error al traducir texto: {e}")
            return text

# ================ GENERADOR DE IMÁGENES ================
class ImageGenerator:
    def __init__(self, stability_api_key, ai_handler):
        self.api_key = stability_api_key
        self.ai_handler = ai_handler
    
    async def generate_image(self, prompt):
        if not self.api_key:
            return None, "No se ha configurado la API key de Stability AI."
        
        try:
            # Traducir el prompt al inglés (Stability API solo acepta inglés)
            prompt_english = await self.ai_handler.translate_text(prompt)
            logger.info(f"Prompt original: {prompt}")
            logger.info(f"Prompt traducido: {prompt_english}")
            
            url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "text_prompts": [{"text": prompt_english}],
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 30,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        image_base64 = data["artifacts"][0]["base64"]
                        image_bytes = base64.b64decode(image_base64)
                        return io.BytesIO(image_bytes), None
                    else:
                        error_text = await response.text()
                        return None, f"Error al generar imagen: {error_text}"
        
        except Exception as e:
            logger.error(f"Error al generar imagen: {e}")
            return None, f"Error: {str(e)}"
    
    async def generate_chart(self, data, title="Estadísticas", x_label="Fecha", y_label="Valor"):
        """Genera un gráfico a partir de datos y devuelve un objeto BytesIO"""
        try:
            # Crear figura y ejes
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Graficar datos
            x = list(range(len(data)))
            ax.bar(x, data, color='skyblue')
            
            # Etiquetas y título
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            
            # Guardar en memoria
            buffer = BytesIO()
            plt.tight_layout()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
        except Exception as e:
            logger.error(f"Error al generar gráfico: {e}")
            return None

# ================ PROMPTS DE SISTEMA ================
INSTRUCCION_NEUTRO = """Responde siempre en español neutro con un tono casual e informal, como si fueras un amigo. No uses regionalismos específicos de ningún país.

Tu tono debe adaptarse al tono del usuario:
- Si el usuario es amable y educado, sé amable y casual pero respetuoso.
- Si el usuario usa un tono más relajado, tú también puedes usarlo.
- Si el usuario usa algunas palabras coloquiales, puedes reflejar ese estilo, pero sin excederte.

Usa expresiones coloquiales de español general que serían entendidas en cualquier país hispanohablante.
No seas excesivamente formal o rígido en ningún caso.

Puedes usar emojis ocasionalmente para expresarte mejor.
Sé directo y ve al grano con tus respuestas, pero mantén un tono conversacional.

Recuerda que eres como un amigo del usuario que está ahí para ayudarlo. No uses fórmulas robóticas como "Como asistente de IA..." o similares.

IMPORTANTE: Siempre proporciona información útil y correcta. Tu tono casual no afecta la calidad de tu información.
"""

# ================ SALUDOS Y RESPUESTAS PREDEFINIDAS ================
SALUDOS_CASUALES = [
    "¡Hola!",
    "¿Qué tal?",
    "¿Cómo estás?",
    "Dime, te escucho",
    "Aquí estoy, ¿qué necesitas?",
    "Hola, ¿en qué puedo ayudarte?",
    "¿En qué te puedo ayudar?",
    "A tus órdenes",
    "Cuéntame",
    "¿Qué necesitas?",
    "Estoy aquí"
]

ERROR_MESSAGES = [
    "Ups, algo salió mal. ¿Podemos intentarlo de nuevo?",
    "Parece que hubo un error. ¿Puedes reformular tu pregunta?",
    "Lo siento, tuve un problema para procesar eso. ¿Intentamos de nuevo?",
    "Algo no funcionó como esperaba. ¿Puedes intentarlo de otra forma?",
    "Ocurrió un error. Dame un momento y vuelve a intentarlo."
]

# ================ CONFIGURACIÓN DEL BOT ================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # Para poder acceder a información de los miembros
bot = commands.Bot(command_prefix='!', intents=intents)

# Instanciar clases principales
db = Database()
bot_memory = BotMemory()
ai_handler = AIHandler(GEMINI_API_KEY)
image_generator = ImageGenerator(STABILITY_API_KEY, ai_handler)

# Canales activos (almacenamiento temporal)
active_channels = {}

# Cooldowns para evitar spam
user_cooldowns = {}
COOLDOWN_TIME = 3  # segundos

# Detectores de intención
IMAGE_KEYWORDS = ["dibuja", "genera imagen", "crea imagen", "haz una imagen", "imagina", "dibujar", "crear imagen"]

# ================ TASKS PERIÓDICAS ================
# Nota: Se ha eliminado la tarea de backup_database, ya que no es útil en Render
# debido al sistema de archivos efímero

# ================ EVENTOS DEL BOT ================
@bot.event
async def on_ready():
    logger.info(f'Bot iniciado como {bot.user}')
    
    # Cargar canales activos desde la base de datos
    active_channel_records = db.get_active_channels()
    for record in active_channel_records:
        guild_id = record["guild_id"]
        channel_id = record["channel_id"]
        active_channels[guild_id] = channel_id
        logger.info(f"Canal activo cargado: Servidor {guild_id}, Canal {channel_id}")
    
    # Sincronizar comandos
    try:
        synced = await bot.tree.sync()
        logger.info(f'Sincronizados {len(synced)} comandos')
    except Exception as e:
        logger.error(f'Error al sincronizar comandos: {e}')
    
    # Establecer estado personalizado
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="tus mensajes | /help"
        )
    )

@bot.event
async def on_message(message):
    # Ignorar mensajes del propio bot
    if message.author == bot.user:
        return
    
    # Registrar mensaje para depuración
    logger.debug(f'Mensaje recibido: {message.channel.id=}, {message.author.id=}, {message.content[:50]}...')
    
    # Verificar si el canal es un canal activo
    is_active = False
    if message.guild and message.guild.id in active_channels:
        is_active = message.channel.id == active_channels[message.guild.id]
    
    # Verificar también en la base de datos (por si se actualizó desde otro lugar)
    if not is_active:
        is_active = db.is_channel_active(message.channel.id)
        # Actualizar la caché si está activo en la base de datos
        if is_active and message.guild:
            active_channels[message.guild.id] = message.channel.id
    
    # Si es un canal activo y no es un comando
    if is_active and not message.content.startswith(('!', '/')):
        # Verificar cooldown para evitar spam
        user_id = message.author.id
        current_time = time.time()
        
        if user_id in user_cooldowns:
            if current_time - user_cooldowns[user_id] < COOLDOWN_TIME:
                # Usuario en cooldown, ignorar mensaje
                return
        
        # Actualizar timestamp de cooldown
        user_cooldowns[user_id] = current_time
        
        # Actualizar interacciones en la base de datos
        db.update_user_interaction(user_id)
        
        # Buscar perfil de usuario
        user_profile = db.get_user_profile(user_id)
        user_name = message.author.display_name
        
        # Si el mensaje es muy corto (posible saludo)
        if len(message.content.strip()) <= 5:
            if user_profile and user_profile["custom_greeting"]:
                # Usar saludo personalizado si existe
                await message.channel.send(user_profile["custom_greeting"].replace("{name}", user_name))
            else:
                # Usar saludo aleatorio
                await message.channel.send(random.choice(SALUDOS_CASUALES))
            return
        
        # Verificar si es una solicitud de imagen
        contenido_lower = message.content.lower()
        
        if any(keyword in contenido_lower for keyword in IMAGE_KEYWORDS):
            async with message.channel.typing():
                await message.channel.send("Voy a generar esa imagen, dame un momento...")
                
                # Generar la imagen (con traducción automática)
                image_bytes, error = await image_generator.generate_image(message.content)
                
                if image_bytes:
                    await message.channel.send(
                        f"Aquí está la imagen que pediste:", 
                        file=discord.File(fp=image_bytes, filename="imagen_generada.png")
                    )
                    # Actualizar estadísticas
                    db.update_daily_stats(images=1, user_id=user_id)
                else:
                    await message.channel.send(f"Lo siento, no pude generar la imagen: {error}")
        else:
            # Respuesta normal de texto
            try:
                async with message.channel.typing():
                    # Preparar contexto con las conversaciones anteriores si están disponibles
                    recent_convos = db.get_recent_conversations(user_id, limit=3)
                    context = ""
                    
                    if recent_convos:
                        context = "Contexto de conversaciones anteriores:\n"
                        for convo in recent_convos:
                            context += f"Usuario: {convo['message']}\nTú: {convo['response']}\n\n"
                    
                    # Crear prompt completo con instrucciones, contexto y mensaje actual
                    prompt_completo = f"{INSTRUCCION_NEUTRO}\n\n{context}Usuario actual ({user_name}): {message.content}"
                    
                    # Obtener respuesta de la IA
                    respuesta = await ai_handler.generate_response(prompt_completo)
                    
                    # Dividir respuestas largas en varios mensajes
                    if len(respuesta) <= 2000:
                        sent_message = await message.channel.send(respuesta)
                    else:
                        first_chunk = respuesta[:2000]
                        sent_message = await message.channel.send(first_chunk)
                        
                        for i in range(2000, len(respuesta), 2000):
                            chunk = respuesta[i:i+2000]
                            await message.channel.send(chunk)
                    
                    # Guardar la conversación en la base de datos
                    db.save_conversation(user_id, message.channel.id, message.content, respuesta)
                    
                    # Actualizar memoria del bot
                    bot_memory.add_interaction(user_id, message.content, respuesta)
                    
                    # Actualizar estadísticas
                    db.update_daily_stats(messages=1, user_id=user_id)
            except Exception as e:
                logger.error(f"Error al procesar mensaje: {e}")
                traceback_str = traceback.format_exc()
                logger.error(traceback_str)
                await message.channel.send(random.choice(ERROR_MESSAGES))
    
    # Procesar comandos del bot
    await bot.process_commands(message)

@bot.event
async def on_guild_join(guild):
    """Evento que se dispara cuando el bot se une a un nuevo servidor"""
    logger.info(f"Bot añadido al servidor: {guild.name} (ID: {guild.id})")
    
    # Buscar un canal de texto para enviar mensaje de bienvenida
    system_channel = guild.system_channel
    if system_channel and system_channel.permissions_for(guild.me).send_messages:
        embed = discord.Embed(
            title="¡Gracias por añadirme!",
            description=(
                "¡Hola! Soy un bot de IA que puede responder preguntas y generar imágenes. "
                "Para empezar, un administrador debe usar el comando `/setchannel` para "
                "elegir el canal donde estaré activo."
            ),
            color=discord.Color.blue()
        )
        embed.add_field(
            name="Comandos principales",
            value=(
                "`/setchannel` - Establece el canal donde responderé automáticamente\n"
                "`/help` - Muestra la lista completa de comandos\n"
                "`/imagen` - Genera una imagen a partir de una descripción"
            )
        )
        await system_channel.send(embed=embed)

# ================ COMANDOS DE SLASH ================
@bot.tree.command(name="setchannel", description="Establece el canal donde el bot responderá automáticamente")
@app_commands.describe(canal="El canal donde quieres que el bot responda")
async def set_channel(interaction: discord.Interaction, canal: discord.TextChannel = None):
    # Verificar permisos
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "Necesitas ser administrador para cambiar esta configuración.",
            ephemeral=True
        )
        return
    
    # Si no se especifica un canal, usar el canal actual
    if canal is None:
        canal = interaction.channel
    
    # Actualizar en base de datos
    db.set_active_channel(interaction.guild.id, canal.id)
    
    # Actualizar en memoria caché
    active_channels[interaction.guild.id] = canal.id
    
    await interaction.response.send_message(
        f"Listo, ahora responderé automáticamente en {canal.mention}",
        ephemeral=True
    )
    logger.info(f'Canal activo configurado: Servidor {interaction.guild.id}, Canal {canal.id}')

@bot.tree.command(name="pregunta", description="Haz una pregunta a la IA")
@app_commands.describe(pregunta="Tu pregunta para la IA")
async def pregunta(interaction: discord.Interaction, pregunta: str):
    await interaction.response.defer()
    
    try:
        user_id = interaction.user.id
        db.update_user_interaction(user_id)
        
        # Obtener respuesta de la IA
        prompt_completo = f"{INSTRUCCION_NEUTRO}\n\nUsuario: {pregunta}"
        respuesta = await ai_handler.generate_response(prompt_completo)
        
        # Guardar conversación
        db.save_conversation(user_id, interaction.channel.id, pregunta, respuesta)
        
        # Dividir respuestas largas
        if len(respuesta) <= 2000:
            await interaction.followup.send(respuesta)
        else:
            mensaje_inicial = respuesta[:2000]
            await interaction.followup.send(mensaje_inicial)
            
            for i in range(2000, len(respuesta), 2000):
                await interaction.followup.send(respuesta[i:i+2000])
        
        # Actualizar estadísticas
        db.update_daily_stats(messages=1, user_id=user_id)
    except Exception as e:
        logger.error(f"Error en comando pregunta: {e}")
        await interaction.followup.send(f"Lo siento, ocurrió un error: {str(e)}")

@bot.tree.command(name="imagen", description="Genera una imagen a partir de una descripción")
@app_commands.describe(descripcion="Descripción de la imagen que quieres generar")
async def imagen(interaction: discord.Interaction, descripcion: str):
    await interaction.response.defer()
    
    user_id = interaction.user.id
    db.update_user_interaction(user_id)
    
    # Generar la imagen
    image_bytes, error = await image_generator.generate_image(descripcion)
    
    if image_bytes:
        await interaction.followup.send(
            f"Aquí está tu imagen:", 
            file=discord.File(fp=image_bytes, filename="imagen_generada.png")
        )
        # Actualizar estadísticas
        db.update_daily_stats(images=1, user_id=user_id)
    else:
        await interaction.followup.send(f"No pude generar la imagen: {error}")

@bot.tree.command(name="perfil", description="Muestra tu perfil de usuario o el de otro usuario")
@app_commands.describe(usuario="Usuario del que quieres ver el perfil (opcional)")
async def perfil(interaction: discord.Interaction, usuario: discord.User = None):
    # Si no se especifica un usuario, usar el que ejecutó el comando
    if usuario is None:
        usuario = interaction.user
    
    user_id = usuario.id
    user_profile = db.get_user_profile(user_id)
    
    if not user_profile:
        await interaction.response.send_message(
            f"No he interactuado lo suficiente con {usuario.display_name} para crear un perfil.",
            ephemeral=True
        )
        return
    
    # Crear un embed para mostrar el perfil
    embed = discord.Embed(
        title=f"Perfil de {usuario.display_name}",
        color=discord.Color.blue()
    )
    
    # Añadir avatar del usuario si está disponible
    if usuario.avatar:
        embed.set_thumbnail(url=usuario.avatar.url)
    
    # Añadir información del perfil
    first_interaction = datetime.datetime.fromisoformat(user_profile["first_interaction"])
    last_interaction = datetime.datetime.fromisoformat(user_profile["last_interaction"])
    
    embed.add_field(
        name="Primera interacción",
        value=first_interaction.strftime("%d/%m/%Y"),
        inline=True
    )
    embed.add_field(
        name="Interacciones totales",
        value=str(user_profile["interaction_count"]),
        inline=True
    )
    embed.add_field(
        name="Última actividad",
        value=last_interaction.strftime("%d/%m/%Y %H:%M"),
        inline=True
    )
    
    # Mostrar saludo personalizado si existe
    if user_profile["custom_greeting"]:
        embed.add_field(
            name="Saludo personalizado",
            value=user_profile["custom_greeting"].replace("{name}", usuario.display_name),
            inline=False
        )
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="set_greeting", description="Establece un saludo personalizado para ti")
@app_commands.describe(saludo="El saludo personalizado (usa {name} para incluir tu nombre)")
async def set_greeting(interaction: discord.Interaction, saludo: str):
    user_id = interaction.user.id
    
    # Verificar longitud del saludo
    if len(saludo) > 200:
        await interaction.response.send_message(
            "El saludo es demasiado largo. Por favor, mantén tu saludo por debajo de 200 caracteres.",
            ephemeral=True
        )
        return
    
    # Guardar saludo en la base de datos
    db.set_custom_greeting(user_id, saludo)
    
    # Confirmar al usuario
    await interaction.response.send_message(
        f"¡Saludo personalizado configurado! Te saludaré con: {saludo.replace('{name}', interaction.user.display_name)}",
        ephemeral=True
    )

@bot.tree.command(name="stats", description="Muestra estadísticas de uso del bot")
@app_commands.describe(dias="Número de días a mostrar (predeterminado: 7)")
async def stats(interaction: discord.Interaction, dias: int = 7):
    # Verificar permisos (solo moderadores o administradores)
    if not interaction.user.guild_permissions.manage_messages and not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "Necesitas permisos de moderador para ver las estadísticas.",
            ephemeral=True
        )
        return
    
    await interaction.response.defer()
    
    # Limitar el número de días
    if dias > 30:
        dias = 30
    
    # Obtener estadísticas de la base de datos
    stats_data = db.get_stats(dias)
    
    if not stats_data:
        await interaction.followup.send("No hay estadísticas disponibles para el período seleccionado.")
        return
    
    # Preparar datos para gráficos
    dates = []
    messages = []
    images = []
    users = []
    
    for stat in stats_data:
        dates.append(stat["date"])
        messages.append(stat["total_messages"])
        images.append(stat["total_images"])
        users.append(stat["active_users"])
    
    # Generar gráfico de mensajes
    messages_chart = await image_generator.generate_chart(
        messages, 
        title="Mensajes procesados",
        x_label="Día",
        y_label="Cantidad"
    )
    
    # Generar gráfico de imágenes
    images_chart = await image_generator.generate_chart(
        images, 
        title="Imágenes generadas",
        x_label="Día",
        y_label="Cantidad"
    )
    
    # Crear resumen textual
    total_messages = sum(messages)
    total_images = sum(images)
    avg_users = sum(users) / len(users) if users else 0
    
    embed = discord.Embed(
        title=f"Estadísticas de los últimos {dias} días",
        color=discord.Color.blue(),
        description=f"Período: {dates[-1]} a {dates[0]}"
    )
    
    embed.add_field(name="Total de mensajes", value=str(total_messages), inline=True)
    embed.add_field(name="Total de imágenes", value=str(total_images), inline=True)
    embed.add_field(name="Usuarios activos (promedio)", value=f"{avg_users:.1f}", inline=True)
    
    # Enviar resumen y gráficos
    await interaction.followup.send(embed=embed)
    
    if messages_chart:
        await interaction.followup.send(
            "Mensajes procesados por día:",
            file=discord.File(fp=messages_chart, filename="mensajes.png")
        )
    
    if images_chart:
        await interaction.followup.send(
            "Imágenes generadas por día:",
            file=discord.File(fp=images_chart, filename="imagenes.png")
        )

@bot.tree.command(name="clear_memory", description="Limpia la memoria de conversaciones contigo")
async def clear_memory(interaction: discord.Interaction):
    user_id = interaction.user.id
    
    # Limpiar memoria en memoria RAM
    bot_memory.clear_user_memory(user_id)
    
    # No podemos eliminar de la base de datos fácilmente, pero podemos marcar como inactivas
    await interaction.response.send_message(
        "He olvidado nuestras conversaciones anteriores. Empezaremos de nuevo.",
        ephemeral=True
    )

@bot.tree.command(name="help", description="Muestra la lista de comandos disponibles")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Comandos disponibles",
        description="Aquí tienes la lista de comandos que puedes usar:",
        color=discord.Color.blue()
    )
    
    # Comandos para todos los usuarios
    embed.add_field(
        name="Comandos generales",
        value=(
            "`/pregunta` - Haz una pregunta directa a la IA\n"
            "`/imagen` - Genera una imagen a partir de una descripción\n"
            "`/perfil` - Muestra tu perfil o el de otro usuario\n"
            "`/set_greeting` - Establece un saludo personalizado\n"
            "`/clear_memory` - Limpia la memoria de conversaciones contigo\n"
            "`/help` - Muestra esta ayuda"
        ),
        inline=False
    )
    
    # Comandos solo para administradores
    embed.add_field(
        name="Comandos de administración",
        value=(
            "`/setchannel` - Establece el canal donde responderé automáticamente\n"
            "`/stats` - Muestra estadísticas de uso del bot"
        ),
        inline=False
    )
    
    # Funcionalidades automáticas
    embed.add_field(
        name="Funcionalidades automáticas",
        value=(
            "• Respondo automáticamente en el canal configurado\n"
            "• Puedes pedirme que dibuje algo con palabras como 'dibuja', 'genera imagen', etc.\n"
            "• Recuerdo el contexto de conversaciones anteriores\n"
            "• Me adapto a tu estilo de comunicación"
        ),
        inline=False
    )
    
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="ping", description="Comprueba la latencia del bot")
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)  # Convertir a milisegundos
    await interaction.response.send_message(f"Pong! Latencia: {latency}ms", ephemeral=True)

@bot.tree.command(name="serverinfo", description="Muestra información sobre el servidor")
async def serverinfo(interaction: discord.Interaction):
    guild = interaction.guild
    
    if not guild:
        await interaction.response.send_message("Este comando solo funciona en servidores.", ephemeral=True)
        return
    
    # Crear embed con información del servidor
    embed = discord.Embed(
        title=f"Información de {guild.name}",
        color=discord.Color.blue()
    )
    
    # Añadir icono del servidor si está disponible
    if guild.icon:
        embed.set_thumbnail(url=guild.icon.url)
    
    # Información general
    embed.add_field(name="ID", value=str(guild.id), inline=True)
    embed.add_field(name="Propietario", value=str(guild.owner), inline=True)
    embed.add_field(name="Creado el", value=guild.created_at.strftime("%d/%m/%Y"), inline=True)
    
    # Estadísticas de miembros
    embed.add_field(name="Miembros", value=str(guild.member_count), inline=True)
    
    # Canales
    text_channels = len(guild.text_channels)
    voice_channels = len(guild.voice_channels)
    categories = len(guild.categories)
    
    embed.add_field(
        name="Canales",
        value=f"📝 Texto: {text_channels}\n🔊 Voz: {voice_channels}\n📁 Categorías: {categories}",
        inline=True
    )
    
    # Roles (limitar a 10 para evitar mensajes demasiado largos)
    roles = guild.roles[-10:] if len(guild.roles) > 10 else guild.roles
    roles.reverse()  # Ordenar de mayor a menor
    roles_str = ", ".join([role.name for role in roles if role.name != "@everyone"])
    
    embed.add_field(
        name=f"Roles ({len(guild.roles) - 1})",
        value=roles_str if roles_str else "Ninguno",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

# ================ MANEJO DE ERRORES ================
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    
    logger.error(f"Error en comando: {error}")
    await ctx.send(f"Ocurrió un error: {str(error)}")

@bot.event
async def on_error(event, *args, **kwargs):
    logger.error(f"Error en evento {event}: {traceback.format_exc()}")

# ================ INICIAR BOT ================
def main():
    try:
        logger.info("Iniciando bot...")
        bot.run(TOKEN)
    except Exception as e:
        logger.critical(f"Error fatal al iniciar el bot: {e}")
    finally:
        # Cerrar conexiones y recursos
        db.close()
        logger.info("Bot finalizado, recursos liberados.")

# ================ SERVIDOR WEB PARA RENDER ================
# Esto es necesario para que Render detecte un puerto abierto y no falle la implementación
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Bot is running!')
    
    def log_message(self, format, *args):
        # Suprimir logs del servidor HTTP
        return

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    logger.info(f"Iniciando servidor web en puerto {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    # Iniciar servidor web en un hilo separado
    if "PORT" in os.environ:  # Solo en Render
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()
        logger.info("Servidor web iniciado en un hilo separado")
    
    # Iniciar el bot
    main()