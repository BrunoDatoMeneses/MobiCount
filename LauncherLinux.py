import subprocess

subprocess.Popen(
    ["python", "RunOnLinux.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)