import os

def check_images(loop_ranges, coordenate_center, register_path="./coordinates.txt"):

    if(not (os.path.exists(register_path))):
        os.mknod(register_path)
        #return False

    with open(register_path, "r") as file:
	    data = file.read()
	    coordinates_register = data.split('\n')

    coordinates_list = []
    
    for latitude in range(0, loop_ranges[0], 1):
        for longitude in range (0, loop_ranges[1], 1):

            coordinate = "{0:.2f}".format(coordenate_center[0] + latitude/100.0) + "," + "{0:.2f}".format(coordenate_center[1] + longitude/100.0)

            coordinates_list.append(coordinate)
    
    print(coordinates_list)