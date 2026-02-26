# we need to add the env variables to docker => access key, secter key, AWS region, docker image name
docker run -e ADMIN_API_KEY=HubtRueBD65daUt4opFc9ez2t -e AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=eu-central-1 -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" -p 8000:8000 class

