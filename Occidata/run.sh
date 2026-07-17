#!/bin/bash

# Options SBATCH :

#SBATCH --job-name=Config_Mobicount    # Nom du Job  
#SBATCH --mail-type=END             # Notification par email de la
#SBATCH --mail-user=bruno.dato@irit.fr     # fin de l'exécution du job.

#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding


#SBATCH --cpus-per-task=4


#SBATCH --output=logs/run_log_%j.out
#SBATCH --error=logs/run_error_%j.err



# Traitement


cd /projects/campmob/




source .venv/bin/activate

cd /projects/campmob/MobiCount/

echo "##### Debut du Job ... #####"

python RunOnOccidata.py

echo "##### Fin du Job.... #####"