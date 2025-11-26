# Autor : Bruno DATO
# Co-autor : Adrien LAMMOGLIA
# Date : 15/11/2025

## ➡️ Step 1 — Install dependencies

import subprocess

process = subprocess.Popen(["python", "--version"], stdout=subprocess.PIPE, text=True)
for line in process.stdout:
    print(line, end="")

process = subprocess.run(["python", "-m","ensurepip","--upgrade"], stdout=subprocess.PIPE, text=True)
for line in process.stdout:
    print(line, end="")

process = subprocess.run(["python", "-m","pip","install","opencv-python"], stdout=subprocess.PIPE, text=True)
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



 


## ➡️ Step 2 — Set project folder, video name and starting hour

PROJECT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/MOBICOUNT/MobiCount"
VIDEO_NAME = "1191553-hd_1920_1080_25fps"
START_HOUR = "14:32:10"  

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

REGION = [(1500, 0), (1500, 3000)]  # VERTICAL LINE
#REGION = [(0, 700), (1920, 700)]    # HORIZONTAL LINE
#REGION = [(860, 0), (860, 1080), (1060, 1080), (1060, 0)]  # VERTICAL RECTANGLE
#REGION = [(760, 0), (760, 1500), (1160, 1500), (1160, 0)]  # THIN VERTICAL RECTANGLE

SHOW_VIDEO = False

CONF = 0.3 # Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.

## ➡️ Step 3 — Create Yolo instance and video writer

import cv2
import csv

from ultralytics import solutions
from datetime import datetime, timedelta
import copy

# Other parameters

VIDEO_FOLDER = PROJECT_FOLDER +"/Video/"
VIDEO_PATH = VIDEO_FOLDER + VIDEO_NAME + ".mp4"
RESULTS_PATH = PROJECT_FOLDER + "/Results/"

# Open the video file

start_time = datetime.strptime(START_HOUR, "%H:%M:%S")
video_path = VIDEO_PATH
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"



# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_writer = cv2.VideoWriter(RESULTS_PATH + VIDEO_NAME +".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
)

results = None

## ➡️ Step 4 — Process the Video


# Process video
print("Processing Video...")

results = None

results_classes_str = str({})
previous_results_classes_str = str({})
results_classes = {}
previous_results_classes = {}

events_list = []

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
                events_list.append([current_time.strftime("%H:%M:%S"),vehicle,"IN",diff_in])

            if diff_out>0:
                events_list.append([current_time.strftime("%H:%M:%S"),vehicle,"OUT",diff_out])

                

    previous_results_classes = copy.deepcopy(results_classes)
    previous_results_classes_str = str(previous_results_classes)

    video_writer.write(results.plot_im)  # write the processed frame.


print("Results: " + str(results.classwise_count))
#TODO compter par pas de temps



## ➡️ Step 5 — Write results (CSV)


with open(RESULTS_PATH + VIDEO_NAME + "_events.csv", "w", newline="") as f:
    
    writer = csv.writer(f)
    writer.writerows(events_list)

with open(RESULTS_PATH + VIDEO_NAME + "_counts.csv", "w", newline="") as f:
    fieldnames = ["TYPE", "IN", "OUT"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for vehicle, counts in results.classwise_count.items():
        row = {"TYPE": vehicle, **counts}
        writer.writerow(row)

print("Counts available at " + RESULTS_PATH + VIDEO_NAME + ".csv")
print("Events available at " + RESULTS_PATH + VIDEO_NAME + ".csv")
print("Video available at " + RESULTS_PATH + VIDEO_NAME +".mp4")

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows