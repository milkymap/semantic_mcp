# base image derivation 
FROM python:3.12-slim 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# initial argument and env 
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV UV_TORCH_BACKEND=cpu

# Node.js environment variables
ENV NVM_DIR=/root/.nvm
ENV NODE_VERSION=20.11.0

# setup required config 
RUN apt update --fix-missing && \
    apt install --yes --no-install-recommends \
         tzdata dialog apt-utils \
         gcc pkg-config git curl build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install NVM and Node.js
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install $NODE_VERSION && \
    nvm alias default $NODE_VERSION && \
    nvm use default

# Add node and npm to path so the commands are available
ENV NODE_PATH=$NVM_DIR/versions/node/v$NODE_VERSION/lib/node_modules
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Verify installations
RUN node --version && npm --version && npx --version

WORKDIR /home/solver
RUN chmod -R g+rwx /home/solver

COPY . ./

# Python dependencies
RUN uv venv && uv sync 

EXPOSE 8000
CMD ["uv", "run", "-m", "src.semantic_mcp.main"]