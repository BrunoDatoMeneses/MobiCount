import DATA
import MobiCount2

FFMPEG_PATH = "ffmpeg" # RAMSES
PROJECT_FOLDER = "/home/adminramses/Documents/MobiCount" # RAMSES

for _videoName, _date, _region in zip(DATA.VIDEO_LIST, DATA.DATES_LIST, DATA.REGION_LIST):

    print(_videoName,_date, _region)
    MobiCount2.count(_videoName, _date, _region, PROJECT_FOLDER, FFMPEG_PATH)


