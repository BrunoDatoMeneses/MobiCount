# Autor : Bruno DATO
# Date : 17/12/2025

## 


import subprocess
import os


## 


FPS = 30
FFMPEG_PATH = "C:/Users/bruno/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
# FFMPEG_PATH = "ffmpeg" # RAMSES
PROJECT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/MOBICOUNT/MobiCount/Results/"
# INPUT_FOLDER = "/home/adminramses/Documents/MobiCount/Results/" # RAMSES

INPUT_FILE = PROJECT_FOLDER+"GX010072_.avi"
OUTPUT_FILE = PROJECT_FOLDER+"GX010072_.mp4"

print("Compressing " + INPUT_FILE)

process = subprocess.Popen([
    FFMPEG_PATH, '-i', INPUT_FILE,
    '-vcodec', 'libx265',           # Codec H265
    '-crf', '30',                   # 28-32 (18=better quality, 51=worse)
    '-preset', 'slow',              # Slower but better compression
    '-vf', 'scale=iw*0.75:ih*0.75', # Rsolution 75%
    '-r', str(FPS),               # FPS reduction
    OUTPUT_FILE,
    '-y'  # Overwrite
],stdout=subprocess.PIPE, text=True)
for line in process.stdout:
    print(line, end="")

#os.remove(RESULTS_PATH + VIDEO_NAME +".avi")

print("Compressed video available at " + OUTPUT_FILE)

