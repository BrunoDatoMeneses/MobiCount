# Options SBATCH :

#SBATCH --job-name=Encode-Simple    # Nom du Job  
#SBATCH --cpus-per-task=4           # Allocation de 4 CPUs par Task

#SBATCH --mail-type=END             # Notification par email de la
#SBATCH --mail-user=bob@irit.fr     # fin de l'exécution du job.

#SBATCH --partition=24CPUNodes

# Traitement

ffmpeg -i video.mp4 -threads $SLURM_CPUS_PER_TASK [...] video.mkv