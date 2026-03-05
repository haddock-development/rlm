FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all server files
COPY rlm_server_v2.py .
COPY repl_environment.py .
COPY safe_builtins.py .
COPY rlm_logger.py .

# Expose port
EXPOSE 5000

# Run server with multiple workers for recursive calls
CMD ["python", "-m", "uvicorn", "rlm_server_v2:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "8"]
