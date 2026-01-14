import subprocess

subprocess.Popen(
    ["python3", "RunOnLinux.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)