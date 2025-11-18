import importlib
import subprocess
import sys

# List of modules to check and their pip names
modules = {
    "cv2": "opencv-python",
    "ultralytics": "ultralytics",
    "csv": None,        # built-in
    "datetime": None,   # built-in
    "copy": None,       # built-in
}

for module, package_name in modules.items():
    try:
        importlib.import_module(module)
        print(f"✅ Module '{module}' is installed.")
    except ImportError:
        if package_name:
            print(f"❌ Module '{module}' is missing. Installing '{package_name}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        else:
            print(f"❌ Module '{module}' is not found, but it's built-in. Check your Python installation.")

print("\n✅ All necessary modules are now installed!")
