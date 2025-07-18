FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 1. Actualizaciones básicas y dependencias para GUI y video
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    tzdata \
    ffmpeg libglib2.0-0 \
    python3-opencv git curl wget unzip \
    && ln -fs /usr/share/zoneinfo/America/Lima /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

    #libsm6 libxext6 libxrender-dev libgl1-mesa-glx 

# 2. Upgrade pip + install jupyterlab
RUN pip install --upgrade pip && pip install jupyterlab

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# 5. Puerto para Jupyter
EXPOSE 8888

# 6. Comando por defecto: iniciar JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
