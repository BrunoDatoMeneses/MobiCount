import subprocess

subprocess.Popen(
    ["python", "RunConfigOnOccidata.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)