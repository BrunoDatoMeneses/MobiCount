# Autor : Bruno DATO
# Date : 17/12/2025

## 


import subprocess
import os


## Lower FPS ffmpeg -i input.mp4 -filter:v "fps=30" output.mp4


FPS = 30
FFMPEG_PATH = "C:/Users/bruno/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
INPUT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/brunodato/MIDOC/Données/Comptage/1/no-audio/"
# FFMPEG_PATH = "ffmpeg" # RAMSES
# INPUT_FOLDER = "/home/adminramses/Documents/MobiCount" # RAMSES
OUTPUT_FOLDER = INPUT_FOLDER + str(FPS) + "FPS/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_exts = (".mp4", ".avi", ".mov", ".mkv")


for filename in os.listdir(INPUT_FOLDER):
    

    if filename.lower().endswith(video_exts):
        in_path = INPUT_FOLDER + filename
        out_path = OUTPUT_FOLDER + str(FPS) + "FPS_" + filename

        print("Input file",in_path)

        process = subprocess.Popen([
            FFMPEG_PATH,
            "-i", in_path,
            "-filter:v", "fps="+str(FPS),
            out_path
        ],stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end="")

        print("Ouput file",out_path)



