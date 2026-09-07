import subprocess
# lancer en tache de fond
# SUR CASU
# /!\ RunOnWindows_2


subprocess.Popen(
    ["python", "RunOnWindows.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)