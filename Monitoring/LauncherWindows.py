import subprocess
# lancer en tache de fond
# SUR CASU
# /!\ RunOnWindows_2


subprocess.Popen(
    ["python", "hw_logger.py","10","monitor.log"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)