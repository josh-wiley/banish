ARG PYTHON_VERSION
FROM mcr.microsoft.com/vscode/devcontainers/python:${PYTHON_VERSION}-bullseye

# poetry
ARG POETRY_VERSION
ENV POETRY_VERSION=${POETRY_VERSION} POETRY_VIRTUALENVS_CREATE=false
RUN su vscode -c "curl -sSL https://install.python-poetry.org | python -" 2>&1

# pytorch
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
