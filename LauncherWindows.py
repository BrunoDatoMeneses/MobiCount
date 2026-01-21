import subprocess

subprocess.Popen(
    ["python", "RunOnWindows.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)