FROM python:3.9-slim

# Hugging Face Spaces requires a non-root user
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Install ffmpeg for librosa
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . /app

# Expose the default port for HF Spaces Docker
EXPOSE 7860

# Run with Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "300", "app:app"]
