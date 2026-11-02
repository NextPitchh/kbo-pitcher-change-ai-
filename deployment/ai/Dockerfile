FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY kbo_bandit/ ./kbo_bandit/
COPY outputs/ ./outputs/
COPY artifact-manifest.json startup.py ./
EXPOSE 8000
CMD ["python", "startup.py"]
