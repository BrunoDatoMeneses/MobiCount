import subprocess

subprocess.Popen(
    ["python3", "RunConfigOnOccidata.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)