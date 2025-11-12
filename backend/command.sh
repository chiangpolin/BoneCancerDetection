docker build -t fastapi-app .
docker run -p 8080:8080 fastapi-app

docker run -ti -v $PWD/backend:/backend --name gcloud-config google/cloud-sdk bash
gcloud auth login
gcloud config set project PROJECT_ID