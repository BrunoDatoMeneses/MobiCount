# Autor : Bruno DATO
# Date : 17/12/2025

## 


import subprocess
import os


## Remove Audio Canal ffmpeg -i input.mp4 -c:v copy -an output.mp4

FFMPEG_PATH = "C:/Users/bruno/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
INPUT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/brunodato/MIDOC/Données/Comptage/1/"
OUTPUT_FOLDER = INPUT_FOLDER +"no-audio/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_exts = (".mp4", ".avi", ".mov", ".mkv")


for filename in os.listdir(INPUT_FOLDER):
    

    if filename.lower().endswith(video_exts):
        in_path = INPUT_FOLDER + filename
        out_path = OUTPUT_FOLDER + "no-audio_"+filename

        print("Input file",in_path)

        process = subprocess.Popen([
            FFMPEG_PATH,
            #"-y",
            "-i", in_path,
            "-c:v", "copy",
            "-an",
            out_path
        ],stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end="")

        print("Ouput file",out_path)



