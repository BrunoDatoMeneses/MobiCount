# Autor : Bruno DATO
# Co-autor : Adrien LAMMOGLIA
# Date : 15/11/2025

## ➡️ Step 1 — Install dependencies

from datetime import datetime, timedelta
import subprocess
import os

DO_INSTALL = False

if DO_INSTALL:

    process = subprocess.Popen(["python", "--version"], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")


    process = subprocess.run(["python", "-m","ensurepip","--upgrade"], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")

    process = subprocess.run(["python", "-m","pip","install","opencv-python"], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")



    process = subprocess.run(["python", "-m","pip","install","python-ffmpeg"], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")

    process = subprocess.run(["python", "-m","pip","install","ultralytics"], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")

    process = subprocess.run(["python", "-m","pip","install","--no-cache-dir","shapely>=2.0.0"], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")

    process = subprocess.run(["python", "-m","pip","install","--no-cache-dir","lap>=0.5.12"], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")




    



print("Install ready")
 


## ➡️ Step 2 — Set project folder, video name and starting hour

FFMPEG_PATH = "C:/Users/bruno/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
PROJECT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/MOBICOUNT/MobiCount"
#VIDEO_NAME = "1191553-hd_1920_1080_25fps"
VIDEO_NAME = "GX010072"
START_DATE_AND_HOUR = datetime(2025, 1, 1, 14, 32, 9)
SECONDS_RANGE = 5 

## ➡️ Step 3 — Set the parameters

CLASSES = [0, 1, 2, 3, 5, 7] # Filters results by class index. For example, classes=[0, 2, 3] only tracks persons, cars and motorcycles.

""" names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  5: bus
  7: truck
 """

#REGION = [(1500, 0), (1500, 3000)]  # VERTICAL LINE 
#REGION = [(1352, 0), (1352, 2028)]  # VERTICAL LINE 2K middle
#REGION = [(676, 0), (676, 2028)]  # VERTICAL LINE 2K first quarter
REGION = [(901, 0), (901, 2028)]  # VERTICAL LINE 2K first tier
#REGION = [(0, 700), (1920, 700)]    # HORIZONTAL LINE
#REGION = [(860, 0), (860, 1080), (1060, 1080), (1060, 0)]  # VERTICAL RECTANGLE
#REGION = [(760, 0), (760, 1500), (1160, 1500), (1160, 0)]  # THIN VERTICAL RECTANGLE

SHOW_VIDEO = False

CONF = 0.3 # Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.





## ➡️ Step 4 — Create Yolo instance and video writer

import cv2
import csv

from ultralytics import solutions

import copy

# Other parameters

VIDEO_FOLDER = PROJECT_FOLDER +"/Video/"
VIDEO_PATH = VIDEO_FOLDER + VIDEO_NAME + ".mp4"
RESULTS_PATH = PROJECT_FOLDER + "/Results/"

# Open the video file

start_time = START_DATE_AND_HOUR



video_path = VIDEO_PATH
cap = cv2.VideoCapture(video_path)


assert cap.isOpened(), "Error reading video file"



# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video Resize
SCALE_FACTOR = 1.0
NEW_WIDTH = int(w * SCALE_FACTOR)
NEW_HEIGHT = int(h * SCALE_FACTOR)
NEW_FPS = fps

# Video Display
DISPLAY_SCALE_FACTOR = 0.3
DISPLAY_WIDTH = int(w * SCALE_FACTOR)
DISPLAY_HEIGHT = int(h * SCALE_FACTOR)
DISPLAY_FPS = fps

codecs = ["avc1", "H264", "XVID", "MJPG"] # General use (avc1 -> .mp4), debug (MJPG -> .avi), Windows (XVID -> .avi), Min size (HEVC but not always installed -> .mkv)
fourcc = cv2.VideoWriter_fourcc(*codecs[3])  # ou "H264", "XVID", "MJPG"
video_writer = cv2.VideoWriter(RESULTS_PATH + VIDEO_NAME +".avi", fourcc, NEW_FPS, (NEW_WIDTH, NEW_HEIGHT))

print("Fps:",fps,"Size:",w,"x",h,"Total frames:",total_frames)





# Initialize object counter object
# https://docs.ultralytics.com/guides/object-counting/#real-world-applications
counter = solutions.ObjectCounter(
    show=SHOW_VIDEO,  # display the output
    region=REGION,  # List of points defining the counting region.
    model="yolo11n.pt",  # Path to Ultralytics YOLO Model File.
    classes=CLASSES,  # Filters results by class index. For example, classes=[0, 2, 3] only tracks the specified classes.
    tracker="bytetrack.yaml",  # Specifies the tracking algorithm to use, e.g., bytetrack.yaml (faster) or botsort.yaml.
    conf = CONF, # Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.
    iou = 0.5, # Sets the Intersection over Union (IoU) threshold for filtering overlapping detections.
    verbose=False,
    figsize=(3.2, 1.8),
    device = "cpu", # Specifies the device for inference (e.g., cpu, cuda:0 or 0). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.
)

results = None

## ➡️ Step 5 — Process the Video

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'
# Process video
print("Processing Video...")

results = None

results_classes_str = str({})
previous_results_classes_str = str({})
results_classes = {}
previous_results_classes = {}


events_dict = {}
counts_by_range = {}
counts_by_range_lists = {}

while cap.isOpened():
    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    elapsed_seconds = frame_index / fps 

    

    if frame_index % (fps) == 0:
        ratio = frame_index/total_frames
        print(str(round(ratio*100, 2)) + " % Frames processed")
        #print(str(timedelta(seconds=elapsed_seconds)) + " Time processed")

    success, im0 = cap.read()

    if not success:
        print("100 % Frames processed")
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)
    current_time = start_time + timedelta(seconds=elapsed_seconds)
    

    results_classes = results.classwise_count
    results_classes_str = str(results_classes)

    if (results_classes_str) != (previous_results_classes_str):

        for vehicle, counts in results_classes.items():
            
            diff_in = 0
            diff_out = 0

            if vehicle in previous_results_classes.keys() :

                diff_in = int(results_classes[vehicle]["IN"])-int(previous_results_classes[vehicle]["IN"])
                diff_out = int(results_classes[vehicle]["OUT"])-int(previous_results_classes[vehicle]["OUT"])

            else:
                diff_in = int(results_classes[vehicle]["IN"])
                diff_out = int(results_classes[vehicle]["OUT"])

            if diff_in>0:

                if vehicle in events_dict.keys() :
                    events_dict[vehicle]["IN"].append([current_time.strftime("%H:%M:%S"),vehicle,"IN",diff_in])
                else:
                    events_dict[vehicle]={"IN":[],"OUT":[]}
                    events_dict[vehicle]["IN"].append([current_time.strftime("%H:%M:%S"),vehicle,"IN",diff_in])



                if vehicle in counts_by_range.keys() :
                    counts_by_range[vehicle]["IN"] += diff_in
                else:
                    counts_by_range[vehicle] = {}
                    counts_by_range[vehicle]["IN"] = diff_in
                    counts_by_range[vehicle]["OUT"] = 0
            

            if diff_out>0:

                if vehicle in events_dict.keys() :
                    events_dict[vehicle]["OUT"].append([current_time.strftime("%H:%M:%S"),vehicle,"OUT",diff_out])
                else:
                    events_dict[vehicle]={"IN":[],"OUT":[]}
                    events_dict[vehicle]["OUT"].append([current_time.strftime("%H:%M:%S"),vehicle,"OUT",diff_out])

                if vehicle in counts_by_range.keys() :
                    counts_by_range[vehicle]["OUT"] += diff_out
                else:
                    counts_by_range[vehicle] = {}
                    counts_by_range[vehicle]["IN"] = 0
                    counts_by_range[vehicle]["OUT"] = diff_out



    if((int(current_time.timestamp()) % SECONDS_RANGE == 0) and (frame_index % fps == 0)):
        
        for _vehicle in counts_by_range.keys():

            if not (_vehicle in counts_by_range_lists):
                counts_by_range_lists[_vehicle]={"IN":[],"OUT":[]}

            if counts_by_range[_vehicle]["IN"] != 0:
                counts_by_range_lists[_vehicle]["IN"].append([current_time.strftime("%H:%M:%S"),counts_by_range[_vehicle]["IN"]])

            if counts_by_range[_vehicle]["OUT"] != 0:
                counts_by_range_lists[_vehicle]["OUT"].append([current_time.strftime("%H:%M:%S"),counts_by_range[_vehicle]["OUT"]])

            
        counts_by_range = {}





    previous_results_classes = copy.deepcopy(results_classes)
    previous_results_classes_str = str(previous_results_classes)

    frame_resized = cv2.resize(results.plot_im, (NEW_WIDTH, NEW_HEIGHT))
    video_writer.write(frame_resized)  # write the processed frame.

    if frame_index == 1:

        cv2.imwrite(RESULTS_PATH+"FirstFrame.jpg", frame_resized)
        print("First frame available at " + RESULTS_PATH + "FirstFrame.jpg")

print(events_dict)
print(counts_by_range_lists)
print("Results: " + str(results.classwise_count))




## ➡️ Step 6 — Write results (CSV)

date_time = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

for _vehicle in counts_by_range_lists.keys():

    for _direction in counts_by_range_lists[_vehicle]:

        with open(RESULTS_PATH + VIDEO_NAME + "_every"+ str(SECONDS_RANGE) + "seconds_" + _vehicle + "_" + _direction + date_time +".csv", "a", newline="") as f:
            
            if counts_by_range_lists[_vehicle][_direction] != 0:
                writer = csv.writer(f)
                writer.writerows(counts_by_range_lists[_vehicle][_direction])


for _vehicle in events_dict.keys():


    for _direction in events_dict[_vehicle]: 

        with open(RESULTS_PATH + VIDEO_NAME + "_events" + date_time + ".csv", "a", newline="") as f:
            
            if events_dict[_vehicle][_direction] != 0:
                writer = csv.writer(f)
                writer.writerows(events_dict[_vehicle][_direction])

        with open(RESULTS_PATH + VIDEO_NAME + "_events_"+ _vehicle + "_" + _direction + date_time +".csv", "a", newline="") as f:
            
            if events_dict[_vehicle][_direction] != 0:
                writer = csv.writer(f)
                writer.writerows(events_dict[_vehicle][_direction])

    

with open(RESULTS_PATH + VIDEO_NAME + "_counts" + date_time + ".csv", "w", newline="") as f:
    fieldnames = ["TYPE", "IN", "OUT"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for vehicle, counts in results.classwise_count.items():
        row = {"TYPE": vehicle, **counts}
        writer.writerow(row)

print("Counts available at " + RESULTS_PATH + VIDEO_NAME + "_counts.csv")
print("Events available at " + RESULTS_PATH + VIDEO_NAME + "_events.csv")
print("Video available at " + RESULTS_PATH + VIDEO_NAME +".avi")

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows

## ➡️ Step 7 — Compress result video 

print("Compressing video...")

output_file = RESULTS_PATH + VIDEO_NAME + ".mp4"

process = subprocess.Popen([
    FFMPEG_PATH, '-i', RESULTS_PATH + VIDEO_NAME +".avi",
    '-vcodec', 'libx265',           # Codec H265
    '-crf', '30',                   # 28-32 (18=better quality, 51=worse)
    '-preset', 'slow',              # Slower but better compression
    '-vf', 'scale=iw*0.75:ih*0.75', # Rsolution 75%
    '-r', str(fps),               # FPS reduction
    output_file,
    '-y'  # Overwrite
],stdout=subprocess.PIPE, text=True)
for line in process.stdout:
    print(line, end="")


#os.remove(RESULTS_PATH + VIDEO_NAME +".avi")

print("Compressed video available at " + output_file)