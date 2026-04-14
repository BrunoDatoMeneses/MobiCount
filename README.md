### Installs

- Install python 3.12 -> https://www.python.org/downloads/release/python-3120/ 
- Other python compatible versions (3.13) 
- Accept every request from python installer
- Install your favorite python IDE 
  - Pycharm -> https://www.jetbrains.com/pycharm/download/?section=windows
- Install ffmpeg
  - Open terminal
    - On windows : winget install ffmpeg
    - On mac : brew install ffmpeg

### Pycharm

- Open Pycharm
- Open working folder Mobicount
- Open MobiCount\MobiCount.ipynb
- Selectec default python interpreter 
  - A message should appear proposing version 3.12
  - If not, add manually python interpreter 3.12
- Wait if automatic updates are running (bottom right of the window)
- Run FlowCounter.ipynb step by step (wait for the notebook package to be installed when running first step)

### FFMPEG

- Remove audio canam: ffmpeg -i input.mp4 -c:v copy -an output.mp4
  - for f in *.mp4; do ffmpeg -i "$f" -c:v copy -an "../no-audio_${f}"; done
- Filter fps: ffmpeg -i input.mp4 -filter:v "fps=30" output.mp4
- Get 30 first seconds: ffmpeg -i input.mp4 -t 30 -c copy output.mp4
- Rename to from *.MP4 to *.mp4 : for f in *.MP4; do mv "$f" "${f%.MP4}.mp4"; done

### Create Python Environement (Linux)

- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
- 
### Create Python Environement (Windows)

- PYTHON_PATH/python.exe -m venv venv
- venv\Scripts\Activate.ps1 (powershell)
- venv\Scripts\activate.bat (cmd)
- source venv/Scripts/activate (Git Bash)
- pip install -r requirements.txt


### Create Python Environement and run on Windows (CMD)

- At MobiCount\..
- PYTHON_PATH\python.exe -m venv .venv
- .venv\Scripts\activate.bat
- pip install -r MobiCount\requirements.txt

### Create Python Environement and run on Linux 

- At MobiCount\..
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r MobiCount\requirements.txt


