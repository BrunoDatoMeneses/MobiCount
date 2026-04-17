import subprocess

subprocess.Popen(
    ["python", "RunOnCasu.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)