# Autor : Bruno DATO
# Date : 07/01/2026


from datetime import datetime, timedelta

VIDEO_LIST = [
              #"no-audio_2GX050072_30s",
              "1191553-hd_1920_1080_25fps",
            #   "no-audio_1GX010072",
            #   "no-audio_1GX020072",
            #   "no-audio_1GX030072",
            #   "no-audio_1GX040072",
            #   "no-audio_1GX050072",
            #   "no-audio_2GX010072",
            #   "no-audio_2GX020072",
            #   "no-audio_2GX030072",
            #   "no-audio_2GX040072",
            #   "no-audio_2GX050072"
              ]

START_DATE_AND_TIME = datetime(2025, 11, 25, 9, 32, 9)
MAX_DURATION = timedelta(minutes=8, seconds=51)

DATES_LIST = [
              #START_DATE_AND_TIME,
              datetime(2025, 1, 1, 14, 32, 9),
            #   START_DATE_AND_TIME,
            #   START_DATE_AND_TIME+MAX_DURATION,
            #   START_DATE_AND_TIME+2*MAX_DURATION,
            #   START_DATE_AND_TIME+3*MAX_DURATION,
            #   START_DATE_AND_TIME+4*MAX_DURATION,
            #   START_DATE_AND_TIME,
            #   START_DATE_AND_TIME+MAX_DURATION,
            #   START_DATE_AND_TIME+2*MAX_DURATION,
            #   START_DATE_AND_TIME+3*MAX_DURATION,
            #   START_DATE_AND_TIME+4*MAX_DURATION,
              ]

REGION_TEST = [(960, 0), (960, 1080)]  # VERTICAL LINE 1080
REGION_1 = [(901, 0), (901, 2028)]  # VERTICAL LINE 2K first tier
REGION_2 = [(1352, 0), (1352, 2028)]  # VERTICAL LINE 2K middle 

REGION_LIST = [
               #REGION_2,
               REGION_TEST,
            #    REGION_1,
            #    REGION_1,
            #    REGION_1,
            #    REGION_1,
            #    REGION_1,
            #    REGION_2,
            #    REGION_2,
            #    REGION_2,
            #    REGION_2,
            #    REGION_2
               ]