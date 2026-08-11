# CleanVoice Studio — Production Docker image
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.cache \
    HF_HOME=/app/.cache

# torch/torchaudio CPU-only (CUDA ছাড়া — ছোট ও সস্তা), তারপর বাকি ডিপেন্ডেন্সি
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .

# আউটপুট ফোল্ডারগুলো নিশ্চিত করা
RUN mkdir -p uploads outputs static

# AI মডেল (~72MB) বিল্ডের সময়ই ডাউনলোড — রানটাইমে লাইভ হবে দ্রুত
RUN python -c "from denoiser import pretrained; pretrained.dns64()" \
 && echo "Model baked ✅"

EXPOSE 7860

# ১টি worker (AI মডেল র‍্যামে একবার লোড হয়) + থ্রেড + লম্বা টাইমআউট (বড় ভিডিও)
CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT:-7860} \
    --workers 1 --threads 8 \
    --timeout 900 --graceful-timeout 900
