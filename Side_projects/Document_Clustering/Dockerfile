
FROM jupyter/minimal-notebook
USER root

RUN sudo -H apt-get update \
    && apt-get install -y \
        git \
        zip \
        sudo \
        nano \
        htop \
        unzip \
        apt-utils \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECT_ROOT /app
ENV CLUSTER_DATA /data
WORKDIR $PROJECT_ROOT

ENV NLTK_DATA=$PROJECT_ROOT/data/nltk_data  
ENV SERVING_PORT=8889

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN sudo -H rm requirements.txt

EXPOSE 8888
EXPOSE 8899
EXPOSE 8999
EXPOSE 9999

RUN cat /etc/*-release

RUN jupyter contrib nbextension install --user
CMD jupyter notebook --ip='0.0.0.0' --port=${SERVING_PORT} --no-browser --allow-root --notebook-dir=${PROJECT_ROOT}
