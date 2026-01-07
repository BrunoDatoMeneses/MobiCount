import subprocess

subprocess.Popen(
    ["python", "MobiCount/RunOnWindows.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)