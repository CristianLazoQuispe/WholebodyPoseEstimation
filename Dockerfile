FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Instalar dependencias del sistema
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 \
#    python3-opencv git curl wget unzip libgtk2.0-dev \
#    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set timezone:
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone


RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 \
    python3-opencv git curl wget unzip \
    && apt-get clean

RUN apt-get update && apt-get install -y \
    libgtk2.0-dev libgl1-mesa-glx libglib2.0-0

# Actualizar pip e instalar JupyterLab
RUN pip install --upgrade pip && pip install jupyterlab

# Crear directorio de trabajo
WORKDIR /workspace

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Exponer el puerto de Jupyter
EXPOSE 8888

# Comando por defecto para iniciar JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

#docker build -f Dockerfile -t wholebody-pose-gpu .
#docker rm wholebody-windows-gpu
#docker run --gpus all --name wholebody-windows-gpu -v $(pwd):/workspace -e DISPLAY=host.docker.internal:0.0 --cpus=8 --memory=20g -p 8888:8888 -it wholebody-pose-gpu