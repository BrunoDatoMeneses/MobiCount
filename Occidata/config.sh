#!/bin/bash

# Options SBATCH :

#SBATCH --job-name=Config_Mobicount    # Nom du Job  
#SBATCH --cpus-per-task=4           # Allocation de 4 CPUs par Task
#SBATCH --mail-type=END             # Notification par email de la
#SBATCH --mail-user=bruno.dato@irit.fr     # fin de l'exécution du job.
#SBATCH --partition=24CPUNodes
#SBATCH --output=logs/log.out
#SBATCH --error=logs/error.err

# Traitement


cd /projects/campmob/

module load ffmpeg/8.0
module load Python/3.12.2

python3 -m venv .venv
source .venv/bin/activate
pip install -r MobiCount/requirements.txt

cd /projects/campmob/MobiCount/

echo "##### Debut du Job ... #####"

python RunConfigOnOccidata.py

echo "##### Fin du Job.... #####"