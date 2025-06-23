FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates texlive-full pandoc

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN useradd --gid root --create-home solver 
WORKDIR /home/solver

COPY . ./
EXPOSE 8000

ENV VIRTUAL_ENV=/home/solver/.venv 
RUN chmod -R g+rwx /home/solver && uv venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv sync 
# entrypoint to launch the engine
ENTRYPOINT ["python", "-m", "src.semantic_mcp"]
CMD ["--help"]