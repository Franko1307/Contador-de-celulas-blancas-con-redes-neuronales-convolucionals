#!/usr/bin/env bash

cd docker

docker build -t imagen_contador_celulas_blancas .

docker create -it --name=contenedor_contador_celulas_blancas --runtime=nvidia imagen_contador_celulas_blancas

docker start contenedor_contador_celulas_blancas

docker cp contenedor_contador_celulas_blancas:/estacion-de-trabajo test_folder/

docker exec -it contenedor_contador_celulas_blancas ./docker-setup.sh
