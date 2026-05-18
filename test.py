from datetime import datetime
import os

def test1():

    dt = datetime(2024, 12, 4, 15, 30, 45)
    total_seconds = dt.timestamp()
    print(total_seconds)  # Ex: 1733328645.0

    print(int(datetime(2024, 12, 4, 15, 30, 45).timestamp())%5)
    print(int(datetime(2024, 12, 4, 15, 30, 44).timestamp())%5)

def test2():

    list1 = [1, 2]
    list2 = [1, 2, 3, 4, 5]


    print(set(list1) )
    print(set(list2) )
    # Méthode avec les sets (la plus efficace)
    additional = list(set(list2) - set(list1))

    print(additional)


def test3():

    folder_path = "C:/Users/bdato/Documents/MobiCount/Video"
    file_names = sorted([os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    print(file_names)

    for _name in file_names:
        print ("____"+str(_name)+"____,")


test3()