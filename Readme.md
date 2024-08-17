## FastAPI starter template for model serving.
Docker, Docker compose, Kubernetes, Helm chart support.
##### packages
    FastAPI 0.112.1 python 3.10 pydantic 2.8.2 poetry 1.6.1
    pytorch 2.1.2 uvicorn 0.30.5

##### As UI - FastAPI OpenAPI

![OpenAPI UI](https://github.com/vpplatonov/test-chart/blob/develop/docs/assets/image_ui.png?raw=true)

###### after press execute

![OpenAPI Res](https://github.com/vpplatonov/test-chart/blob/develop/docs/assets/image_response.png?raw=true)

### Install Helm
```sh
brew install helm
helm version
```
### Creating Your Own Charts
```sh
helm create test-chart
```
##### validate that it is well-formed by running
```sh
helm lint
```
#### package the chart up for distribution, you can run the command:
```sh
helm package test-chart
```
#### that chart can now easily be installed by
```sh
helm install test-task ./test-chart-0.1.0.tgz
```
#### for debug templates
```sh
helm install --debug --dry-run test-task ./test-chart
```
#### Convert docker-compose to kubernetes yaml https://kubernetes.io/docs/tasks/configure-pod-container/translate-compose-kubernetes/
```sh    
brew install kompose
kompose convert
```
#### Convert kubernetes yaml to helm chart use helmify (update xCode first)
```sh
brew install arttor/tap/helmify
helmify -f kubernetes test-chart
helm lint test-chart
```
### RUN
#### cli (in browser http://localhost:8000/docs)
```sh
poetry install
python main.py
```
#### from docker (in browser http://localhost:8000/docs)
```sh
docker-compose -f docker-compose.yml up -d
```
#### from kubernetis
```sh
helm install test-task ./test-chart-0.1.0.tgz
```
###### create alias in /etc/hosts for cluster.local domain
    127.0.0.1 cluster.local localhost
    http://cluster.local:8000/docs
```shell
kubectl get pods
kubectl port-forward test-task-test-chart-api-<container-id> 8000:8000
```