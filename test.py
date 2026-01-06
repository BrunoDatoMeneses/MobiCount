from datetime import datetime

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


test2()