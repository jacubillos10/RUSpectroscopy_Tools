#!/bin/bash
# ###### Zona de ParÃ¡metros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=ElasticityTensorJob
#SBATCH -p short                        #Cola a usar, Default=short (Ver colas y lÃ­mites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                            #Nodos requeridos, Default=1
#SBATCH -n 1                            #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=16              #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=2048              #Memoria en Mb por CPU, Default=2048
#SBATCH --time=24:03:00                 #Tiempo mÃ¡ximo de corrida, Default=2 horas
#SBATCH --mail-user=ja.cubillos10@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o logs/ElasticityTensor_job_%j.log                   #Nombre de archivo de salida
#
########################################################################################

module load anaconda/python3.9
source mi_venv/bin/activate

python3 generate_data.py
