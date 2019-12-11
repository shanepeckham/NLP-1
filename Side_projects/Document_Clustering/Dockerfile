
FROM jupyter/pyspark-notebook
# FROM pytorch/pytorch

USER root

RUN sudo apt-get update \
    && sudo apt-get install -y \
        git \
        zip \
        nano \
        htop \
        unzip \
    && rm -rf /var/lib/apt/lists/*


ENV PROJECT_ROOT /home/jovyan/work
# ENV PROJECT_ROOT /app
WORKDIR $PROJECT_ROOT

ENV NLTK_DATA=$PROJECT_ROOT/data/nltk_data  

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN rm requirements.txt

# COPY . .

EXPOSE 8888
EXPOSE 8899

RUN cat /etc/*-release

# CMD ["bin". "bash"]
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
