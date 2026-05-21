# Autor : Bruno DATO
# Date : 07/01/2026


from datetime import datetime, timedelta

VIDEO_LIST = [
            "no-audio_7.3_06052026_07_32__TEST"
              ]

START_DATE_AND_TIME = datetime(2025, 11, 25, 9, 32, 9)
MAX_DURATION = timedelta(minutes=8, seconds=52)

DATES_LIST = [

            START_DATE_AND_TIME # a modifier
             ]

REGION_TEST = [(960, 0), (960, 1080)]  # VERTICAL LINE 1080
REGION_1 = [(901, 0), (901, 2028)]  # VERTICAL LINE 2K first tier
REGION_2 = [(1352, 0), (1352, 2028)]  # VERTICAL LINE 2K middle 

REGION_LIST = [
              REGION_TEST
              ]


MODELS_LIST = [
              "yolo11l"
              ]