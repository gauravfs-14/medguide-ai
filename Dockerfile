FROM python:3.11-slim

WORKDIR /medguide-ai

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    software-properties-common \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the repo
RUN git clone https://github.com/gauravfs-14/medguide-ai . 

# Create virtual environment
RUN python -m venv .venv

# Install Python dependencies from pyproject.toml
COPY pyproject.toml .
RUN .venv/bin/pip install --upgrade pip setuptools wheel && \
    .venv/bin/pip install .

# Streamlit default port
EXPOSE 8501

# Healthcheck (optional but useful)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch app
ENTRYPOINT [".venv/bin/streamlit", "run", "mcp-client/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
