# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto (si es necesario)
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "bot.py"]