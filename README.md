# 1.Similar to actor

RUN (Streamlit)
```
streamlit run main.py
```

- Docker
```
- Собрать образ: docker build -t your-name-image .
- Посмотреть все собранные образы: docker images
- Удалить Docker образ: docker rmi your-id-image

- Собрать приложение из Docker image (контейнер): docker run your-name-image
- Если хотим запустить конкретный, например, скрипт внутри образа: docker run your-name-image python train.py

- Посмотреть все запущенные контейнеры: docker ps
- Посмотреть все запущенные/не запущенные контейнеры: docker ps -a

- Остановить запущенный определенный контейнер: docker stop my_container
- Остановить все запущенные контейнеры (если они есть): docker stop $(docker ps -a -q)

- Удалить все контейнеры (если они есть): docker container rm $(docker ps -a -q)
```

- Mlflow
```
pip install mlflow

mkdir mlflow
export MLFLOW_REGISTRY_URI=mlflow

Запуск сервера: mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///${MLFLOW_REGISTRY_URI}/mlflow.db --default-artifact-root ${MLFLOW_REGISTRY_URI}
```

- Airflow
```
pip install apache-airflow==2.8.1 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.8.txt"

airflow db init

airflow.cfg прописать:
[webserver] rbac = True
load_examples = False

airflow users create --username geramond --firstname Maksim --lastname Fomin --role Admin --email geramond@gmail.com

airflow webserver -p 8080
airflow scheduler
```

- FastAPI
```
python3 -m uvicorn main:app --host=127.0.0.1 --port 8000 --reload
```
