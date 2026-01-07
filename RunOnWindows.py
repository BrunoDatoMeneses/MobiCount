import DATA
import MobiCount2

FFMPEG_PATH = "C:/Users/bruno/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe"
PROJECT_FOLDER = "C:/Users/bruno/OneDrive/Documents/Repositories/MOBICOUNT/MobiCount"

for _videoName, _date, _region in zip(DATA.VIDEO_LIST, DATA.DATES_LIST, DATA.REGION_LIST):

    print(_videoName,_date, _region)
    MobiCount2.count(_videoName, _date, _region, PROJECT_FOLDER, FFMPEG_PATH)


