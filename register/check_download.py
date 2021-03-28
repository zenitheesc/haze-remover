import os
from .already_used import coordinates_register

def check_images(loop_ranges, coordinate_center, images_path, register_path="./coordinates.txt"):

    coordinates_already_used = coordinates_register(register_path)

    coordinates_list = []

    for latitude in range(0, loop_ranges[0], 1):
        for longitude in range (0, loop_ranges[1], 1):

            coordinate = "{0:.2f}".format(coordinate_center[0] + latitude/100.0) + "," + "{0:.2f}".format(coordinate_center[1] + longitude/100.0)
            
            file_total_path = images_path + coordinate + ".png"

            if(not(os.path.exists(file_total_path)) and not(coordinate in coordinates_already_used)):
                coordinates_list.append(coordinate)
            else:
                print("Coordinate " + coordinate + " already used")
    
    print(coordinates_list)
    return coordinates_list