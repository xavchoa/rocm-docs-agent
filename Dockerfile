FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md config.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# The index is built outside the image and mounted at /app/data at run time.
EXPOSE 8000
CMD ["uvicorn", "rocm_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
