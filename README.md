# Tutorial docker for Data Science project

- Установка Docker
https://docs.docker.com/get-docker/

- Собрать образ
```
docker build -t your-name-image .
```

- Посмотреть все собранные образы
```
docker images
```

- Удалить Docker образ
```
docker rmi your-id-image
```

- Собрать приложение из Docker image (контейнер)
```
docker run your-name-image
```

- Если хотим запустить конкретный, например, скрипт внутри образа
```
docker run your-name-image python train.py
```

- Посмотреть все запущенные контейнеры
```
docker ps   
```

- Посмотреть все запущенные/не запущенные контейнеры
```
docker ps -a
```

- Остановить запущенный определенный контейнер
```
docker stop my_container
```

- Остановить все запущенные контейнеры (если они есть)
```
docker stop $(docker ps -a -q)
```

- Удалить все контейнеры (если они есть)
```
docker container rm $(docker ps -a -q)
```
