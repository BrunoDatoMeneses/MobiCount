import subprocess

subprocess.Popen(
    ["python", "MobiCount/MobiCountXP.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)