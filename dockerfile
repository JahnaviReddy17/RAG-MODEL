FROM python:3.11-slim

# Docker Spaces run as UID 1000 — create a matching user to avoid permission issues
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Docker Spaces expect the app on port 7860 by default
EXPOSE 7860

CMD ["streamlit", "run", "ui.py", "--server.port=7860", "--server.address=0.0.0.0"]
