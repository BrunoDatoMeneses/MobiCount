import subprocess

subprocess.Popen(
    ["python", "MobiCountXP.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)