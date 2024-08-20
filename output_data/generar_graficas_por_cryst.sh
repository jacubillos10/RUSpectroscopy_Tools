#!/bin/bash

for (( i = 0; i<4; i++ ))
do
	python3 generar_graficas_mse.py $i
done
