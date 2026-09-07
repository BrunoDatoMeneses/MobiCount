import DATA
import MobiCount2
import logging
from datetime import datetime
import os
# SUR CASU
# /!/ RunOnWindows_2

logger = logging.getLogger(__name__)

FFMPEG_PATH = "C:/Users/bruno/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
PROJECT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/MOBICOUNT/MobiCount"
DATE = datetime.now().strftime("%Y%m%d")
DATE_TIME =  datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_FOLDER = PROJECT_FOLDER + "/Config/" + DATE + "/"

os.makedirs(RESULTS_FOLDER, exist_ok=True)
logging.basicConfig(filename=RESULTS_FOLDER +'/'+ DATE_TIME + '.log', encoding='utf-8', level=logging.DEBUG)

MobiCount2.runSeveralConfigs(DATA.VIDEO_LIST, DATA.DATES_LIST, DATA.REGION_LIST, DATA.MODELS_LIST, PROJECT_FOLDER, FFMPEG_PATH, DATE, DATE_TIME, RESULTS_FOLDER)



