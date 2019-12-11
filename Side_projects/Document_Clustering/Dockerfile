
FROM jupyter/pyspark-notebook
# FROM pytorch/pytorch

RUN apt update \
    && apt install -y \
        git \
        zip \
        nano \
        htop \
        unzip \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECT_ROOT /app
WORKDIR $PROJECT_ROOT

ENV NLTK_DATA=/app/data/nltk_data  

COPY README.md ./
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN rm requirements.txt

# COPY . .

EXPOSE 8888

RUN cat README.md
RUN cat /etc/*-release

# RUN sed -i -e 's/\r$//' launch.sh
# CMD ["./launch.sh"]
CMD ["bin". "bash"]
# CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
