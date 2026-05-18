import DATA
import MobiCount2
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

FFMPEG_PATH = "C:/Users/bdato/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe"
PROJECT_FOLDER = "C:/Users/bdato/Documents/MobiCount"
DATE = datetime.now().strftime("%Y%m%d")
DATE_TIME =  datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FOLDER = PROJECT_FOLDER + "/Config/" + DATE + "/"

os.makedirs(RESULTS_FOLDER, exist_ok=True)
logging.basicConfig(filename=RESULTS_FOLDER +'/'+ DATE_TIME + '.log', encoding='utf-8', level=logging.DEBUG)

MobiCount2.runSeveralCounts(DATA.VIDEO_LIST, DATA.DATES_LIST, DATA.REGION_LIST, DATA.MODELS_LIST, PROJECT_FOLDER, FFMPEG_PATH, DATE, DATE_TIME, RESULTS_FOLDER)


