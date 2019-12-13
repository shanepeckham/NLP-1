
# Preparing environment in Host
# ------------------------------

# SSH into your machine
# ssh -i <path_to_your_key> user@host

export SSH_USER=pablo
export SSH_HOST=10.99.195.149

export KEY_PATH="/mnt/c/Users/RUIZP4/Documents/DOCS/RnD/id_rsa_BASF_RnD"
export LOCAL_PROJECT_PATH="/mnt/c/Users/RUIZP4/Documents/DOCS/Pablo_Personal/NLP_Projects/Side_projects/Document_Clustering"
export REMOTE_PROJECT_PATH="/home/pablo/Side_NLP_Tests/"
export EXCLUDE_SYNC_FILE="exclude_sync.txt"

ssh -i ${KEY_PATH} ${SSH_USER}@${SSH_HOST}

# Open LocalForwarding on the Background
ssh -i $KEY_PATH \
    -fNL 8889:127.0.0.1:18889 \
    -fNL 8899:127.0.0.1:18899 \
    -fNL 8999:127.0.0.1:18999 \
    -fNL 9999:127.0.0.1:19999 \
    pablo@10.99.195.149

# Copy the data from the datadrive into your folder
# cp -r /datadrive/Projects/knowledge_dashboard /home/<user>/<project_folder>

cp -r /datadrive/Projects/knowledge_dashboard /home/pablo/NLP/data

# Run bulk unzip notebook

rsync -auv -e "ssh -i ${KEY_PATH}" \
    --exclude-from=${EXCLUDE_SYNC_FILE} \
    $LOCAL_PROJECT_PATH pablo@10.99.195.149:$REMOTE_PROJECT_PATH

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

export DOCKER_PORT=8889
export CLUSTER_PORT=18889

export DOCKER_IMAGE=pablorr10/nlp:minimal
export CONTAINER_NAME=nlpminimal

export CLUSTER_ROOT=/home/pablo/Side_NLP_Tests/Document_Clustering
export CONTAINER_ROOT=/app

export CLUSTER_DATA=/datadrive/madrid
export CONTAINER_DATA=${CONTAINER_ROOT}/globaldata

# Open a shell in the container
docker stop ${CONTAINER_NAME} || true
docker run --rm -dit \
    --name ${CONTAINER_NAME} \
    -p ${CLUSTER_PORT}:${DOCKER_PORT} \
    -v ${CLUSTER_ROOT}:${CONTAINER_ROOT} \
    -v ${CLUSTER_DATA}:${CONTAINER_DATA} \
    ${DOCKER_IMAGE}

docker exec -it ${CONTAINER_NAME} bash

docker logs ${CONTAINER_NAME}
