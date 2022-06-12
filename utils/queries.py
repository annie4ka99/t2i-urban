import csv


locations = {}
with open('D:/Bachelor/t2i-urban/locations.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        locations[row[0]] = row[1]


def get_query(location_name):
    if location_name in locations:
        return locations[location_name]
    else:
        return location_name
