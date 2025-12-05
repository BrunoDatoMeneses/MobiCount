from datetime import datetime

dt = datetime(2024, 12, 4, 15, 30, 45)
total_seconds = dt.timestamp()
print(total_seconds)  # Ex: 1733328645.0

print(int(datetime(2024, 12, 4, 15, 30, 45).timestamp())%5)
print(int(datetime(2024, 12, 4, 15, 30, 44).timestamp())%5)