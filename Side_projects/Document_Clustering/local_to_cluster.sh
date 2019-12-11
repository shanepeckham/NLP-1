
# Preparing environment in Host
# ------------------------------

# SSH into your machine
# ssh -i <path_to_your_key> <open_local_forwarding> user@host
ssh -i /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD \
    -L 8899:127.0.0.1:8899 \
    pablo@10.99.195.149

# Copy the data from the datadrive into your folder
# cp -r /datadrive/Projects/knowledge_dashboard /home/<user>/<project_folder>
cp -r /datadrive/Projects/knowledge_dashboard /home/pablo/NLP/data

# Run bulk unzip notebook

rsync -auv \
    -e "ssh -i /mnt/c/Users/RUIZP4/Documents/DOCS/Pablo_Personal/NLP_Projects" \
    --exclude-from="exclude_sync.txt" \
    /mnt/c/Users/RUIZP4/Documents/DOCS/Pablo_Personal/NLP_Projects \
    pablo@10.99.195.149:/home/pablo/Side_NLP_Tests/


# Move Corpus
scp -i /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD \
    -rp /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/NLP/data/catalog \
    pablo@10.99.195.149:/home/pablo/NLP/data

# Move Stopwords
scp -i /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD \
    /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/NLP/data/stopwords.pkl \
    pablo@10.99.195.149:/home/pablo/NLP/data

# Move Spacy Language Model
scp -i /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD \
    -rp /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/NLP/data/lang_models \
    pablo@10.99.195.149:/home/pablo/NLP/data

# Move NLTK WordNet Data
scp -i /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD \
    -rp /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/NLP/data/nltk_data \
    pablo@10.99.195.149:/home/pablo/NLP/data


# Move from Remote to Host --> (Notebooks develped in remote bring to host to add it to Git)
scp -i /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD \
    -r pablo@10.99.195.149:/home/pablo/NLP/jupyter_notebooks \
    /mnt/c/Users/RUIZP4/Documents/DOCS/RnD/NLP/jupyter_notebooks/



# Using Docker in Host
# ---------------------

docker pull pablorr10/rnd:dev

# Open a shell in the container
docker run --rm -it \
    --name nlpshell \
    -v ${PWD}:/app \
    pablorr10/rnd:dev \
    bash

# Run detached jupyter notebook
docker run --rm -d \
    --name nlpnotebook \
    -p 8899:8888 \
    -v ${PWD}:/app \
    pablorr10/nlp:dev 

docker logs nlpnotebook


# Install NLTK !
nltk.download('stopwords')
