# Autor : Bruno DATO
# Co-autor : Adrien LAMMOGLIA
# Date : 15/11/2025

from datetime import datetime, timedelta
import subprocess
import os
import cv2
import csv
from ultralytics import solutions
import copy
import logging


VERBOSE = True

logger = logging.getLogger(__name__)

def log(*args):


    if VERBOSE:

        message = datetime.now().strftime("[%Y%m%d_%H:%M:%S]") + " "
        for _arg in args:
            message += str(_arg) + " "

        #print(message)
        logger.info(message)


def runSeveralCounts(VIDEO_LIST, DATES_LIST, REGION_LIST, MODEL_LIST, PROJECT_FOLDER, FFMPEG_PATH, DATE, DATE_TIME, RESULTS_FOLDER):

    logger.info(str(datetime.now().strftime("[%Y%m%d_%H:%M:%S]")) +" "+ str(VIDEO_LIST) +" "+ str(DATES_LIST) +" "+ str(REGION_LIST) +" "+ str(MODEL_LIST))

    for _videoName, _date, _region, _model in zip(VIDEO_LIST, DATES_LIST, REGION_LIST, MODEL_LIST):

        logger.info(str(datetime.now().strftime("[%Y%m%d_%H:%M:%S]")) +" "+ str(_videoName) +" "+ str(_date) +" "+ str(_region) +" "+ str(_model))
        MobiCount2.count(_videoName, _date, _region, PROJECT_FOLDER, FFMPEG_PATH, DATE, DATE_TIME, RESULTS_FOLDER, _model)



def count(VIDEO_NAME, START_DATE_AND_HOUR, REGION, PROJECT_FOLDER, FFMPEG_PATH, DATE, DATE_TIME, RESULTS_PATH, MODEL):

    ## ➡️ Step 1 — Install dependencies

    

    DO_INSTALL = False

    if DO_INSTALL:

        process = subprocess.Popen(["python", "--version"], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            log(line, end="")


        process = subprocess.run(["python", "-m","ensurepip","--upgrade"], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            log(line, end="")

        process = subprocess.run(["python", "-m","pip","install","opencv-python"], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            log(line, end="")



        process = subprocess.run(["python", "-m","pip","install","python-ffmpeg"], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            log(line, end="")

        process = subprocess.run(["python", "-m","pip","install","ultralytics"], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            log(line, end="")

        process = subprocess.run(["python", "-m","pip","install","--no-cache-dir","shapely>=2.0.0"], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            log(line, end="")

        process = subprocess.run(["python", "-m","pip","install","--no-cache-dir","lap>=0.5.12"], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            log(line, end="")




        



    log("Install ready")
    


    ## ➡️ Step 2 — Set project folder, video name and starting hour

    # FFMPEG_PATH = "C:/Users/bruno/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
    # PROJECT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/MOBICOUNT/MobiCount"
    # FFMPEG_PATH = "ffmpeg" # RAMSES
    # PROJECT_FOLDER = "/home/adminramses/Documents/MobiCount" # RAMSES

    #VIDEO_NAME = "1191553-hd_1920_1080_25fps"
    #START_DATE_AND_HOUR = datetime(2025, 1, 1, 14, 32, 9)


    ## ➡️ Step 3 — Set the parameters

    # CLASSES = [0, 1, 2, 3, 5, 7] # Filters results by class index. For example, classes=[0, 2, 3] only tracks persons, cars and motorcycles.
    # CLASSES_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

    CLASSES = [0, 1, 2, 3, 5] # Filters results by class index. For example, classes=[0, 2, 3] only tracks persons, cars and motorcycles.
    CLASSES_NAMES = ["person", "bicycle", "car", "motorcycle", "bus"]

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
    #REGION = [(901, 0), (901, 2028)]  # VERTICAL LINE 2K first tier
    #REGION = [(0, 700), (1920, 700)]    # HORIZONTAL LINE
    #REGION = [(860, 0), (860, 1080), (1060, 1080), (1060, 0)]  # VERTICAL RECTANGLE
    #REGION = [(760, 0), (760, 1500), (1160, 1500), (1160, 0)]  # THIN VERTICAL RECTANGLE

    SHOW_VIDEO = False

    CONF = 0.1 # Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.





    ## ➡️ Step 4 — Create Yolo instance and video writer

    

    # Other parameters

    date = DATE
    date_time =  DATE_TIME

    

    VIDEO_FOLDER = PROJECT_FOLDER +"/Video/"
    VIDEO_PATH = VIDEO_FOLDER + VIDEO_NAME + ".mp4"
    #RESULTS_PATH = PROJECT_FOLDER + "/Results/" + date + "/"

    

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
    video_writer = cv2.VideoWriter(RESULTS_PATH + date_time + "__" + VIDEO_NAME +".avi", fourcc, NEW_FPS, (NEW_WIDTH, NEW_HEIGHT))

    log("Fps:",fps,"Size:",w,"x",h,"Total frames:",total_frames)





    # Initialize object counter object
    # https://docs.ultralytics.com/guides/object-counting/#real-world-applications
    counter = solutions.ObjectCounter(
        show=SHOW_VIDEO,  # display the output
        region=REGION,  # List of points defining the counting region.
        model=MODEL,  # Path to Ultralytics YOLO Model File.
        classes=CLASSES,  # Filters results by class index. For example, classes=[0, 2, 3] only tracks the specified classes.
        tracker="botsort.yaml",  # Specifies the tracking algorithm to use, e.g., bytetrack.yaml (faster) or botsort.yaml.
        conf = CONF, # Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.
        iou = 0.9, # Sets the Intersection over Union (IoU) threshold for filtering overlapping detections.
        verbose=False,
        figsize=(3.2, 1.8),
        blur_ratio=0.5,
        max_hist = 5,
        device = "cpu", # Specifies the device for inference (e.g., cpu, cuda:0 or 0). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.
    )



    results = None

    ## ➡️ Step 5 — Process the Video

    log(f"Ultralytics Solutions: ✅ {counter.CFG}")

    os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'
    # Process video
    log("Processing Video...")

    results = None

    results_classes_str = str({})
    previous_results_classes_str = str({})
    results_classes = {}
    previous_results_classes = {}
    object_ids = {}
    previous_object_ids = {}


    events_dict = {}
    #counts_by_range = {}
    #counts_by_range_lists = {}

    while cap.isOpened():
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elapsed_seconds = frame_index / fps 

        

        if frame_index % (fps) == 0:
            ratio = frame_index/total_frames
            log(str(round(ratio*100, 2)) + " % Frames processed")
            #log(str(timedelta(seconds=elapsed_seconds)) + " Time processed")

        success, im0 = cap.read()

        if not success:
            log("100 % Frames processed")
            log("Video frame is empty or processing is complete.")
            break

        results = counter(im0)
        current_time = start_time + timedelta(seconds=elapsed_seconds)
        

        results_classes = results.classwise_count
        results_classes_str = str(results_classes)
        
        
        #log(counter.counted_ids)
        

        object_ids = counter.counted_ids

        new_object_ids = list(set(object_ids) - set(previous_object_ids))

        if (results_classes_str) != (previous_results_classes_str):

            event_dict = {}

            event_dict["timeStamp"]=current_time.strftime("%H:%M:%S")
            event_dict["ids"]=new_object_ids
            

            for vehicle, counts in results_classes.items():
                
                diff_in = 0
                diff_out = 0

                if vehicle in previous_results_classes.keys() :

                    diff_in = int(results_classes[vehicle]["IN"])-int(previous_results_classes[vehicle]["IN"])
                    diff_out = int(results_classes[vehicle]["OUT"])-int(previous_results_classes[vehicle]["OUT"])

                else:
                    diff_in = int(results_classes[vehicle]["IN"])
                    diff_out = int(results_classes[vehicle]["OUT"])

                
                event_dict[vehicle]={"IN":0,"OUT":0}

                if diff_in>0:

                    event_dict[vehicle]["IN"] = diff_in

                

                if diff_out>0:

                    event_dict[vehicle]["OUT"] = diff_out


                    
            events_dict[str(frame_index)] = event_dict

            #log(event_dict)
            







        previous_object_ids = copy.deepcopy(object_ids)
        previous_results_classes = copy.deepcopy(results_classes)
        previous_results_classes_str = str(previous_results_classes)

        frame_resized = cv2.resize(results.plot_im, (NEW_WIDTH, NEW_HEIGHT))
        video_writer.write(frame_resized)  # write the processed frame.

        if frame_index == 1:

            cv2.imwrite(RESULTS_PATH + date_time + "__" + VIDEO_NAME+"_FirstFrame.jpg", frame_resized)
            log("First frame available at " + RESULTS_PATH + date_time + "__" + VIDEO_NAME+"_FirstFrame.jpg")

    #log(events_dict)
    #log(counts_by_range_lists)
    log("Results: " + str(results.classwise_count))




    ## ➡️ Step 6 — Write results (CSV)

    


    with open(RESULTS_PATH + date_time + "__" + VIDEO_NAME + "_events"  + ".csv", "a", newline="") as f:

        fieldnames = ["TIME", "FRAME"]
        for _vehicle in CLASSES_NAMES:
            fieldnames += [_vehicle + " IN", _vehicle + " OUT"]

        fieldnames += ["IDS"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _frameIndex, _eventDict in events_dict.items():

            #log(_eventDict)

            rowForCSV = [_eventDict["timeStamp"],  _frameIndex]

            for _vehicle in CLASSES_NAMES:


                if _vehicle in _eventDict.keys():
                    rowForCSV = rowForCSV + [_eventDict[_vehicle]["IN"], _eventDict[_vehicle]["OUT"]]
                else:
                    rowForCSV = rowForCSV + [0, 0]


            rowForCSV = rowForCSV + [_eventDict["ids"]]
            #log(rowForCSV)
        
        
            writer = csv.writer(f)
            writer.writerow(rowForCSV)



        

    with open(RESULTS_PATH + date_time + "__" + VIDEO_NAME + "_counts"  + ".csv", "w", newline="") as f:
        fieldnames = ["TYPE", "IN", "OUT"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for vehicle, counts in results.classwise_count.items():
            row = {"TYPE": vehicle, **counts}
            writer.writerow(row)

    log("Counts available at " + RESULTS_PATH + date_time + "__" + VIDEO_NAME + "_counts.csv")
    log("Events available at " + RESULTS_PATH + date_time + "__" + VIDEO_NAME + "_events.csv")
    log("Video available at " + RESULTS_PATH + date_time + "__" + VIDEO_NAME +".avi")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # destroy all opened windows

    ## ➡️ Step 7 — Compress result video 

    log("Compressing video...")

    output_file = RESULTS_PATH + date_time + "__" +  VIDEO_NAME + ".mp4"

    process = subprocess.run([
        FFMPEG_PATH, '-i', RESULTS_PATH + date_time + "__" + VIDEO_NAME +".avi",
        '-vcodec', 'libx265',           # Codec H265
        '-crf', '30',                   # 28-32 (18=better quality, 51=worse)
        '-preset', 'slow',              # Slower but better compression
        '-vf', 'scale=iw*0.5:ih*0.5', # Rsolution 50%
        '-r', str(fps/2),               # FPS reduction
        output_file,
        '-y'  # Overwrite
    ], 
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,)

    """if VERBOSE:
        for line in process.stdout:
            log(line, end="")"""


    #os.remove(RESULTS_PATH + date_time + "__" + VIDEO_NAME +".avi")

    log("Compressed video available at " + output_file)






