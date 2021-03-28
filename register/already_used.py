import os

def coordinates_register(register_path="./coordinates.txt"):
    
    files_register = files_already_used(register_path)
    
    if not files_register: 
        return False
    
    coordinates_already_used = []

    for file in files_register:
        no_extension = file.split(".png")[0]

        final = no_extension.split("_")
            
        coordinate = final[len(final) - 1]

        if(not coordinate in coordinates_already_used):
            coordinates_already_used.append(coordinate)

    return coordinates_already_used

def files_already_used(register_path="./coordinates.txt"):

    if(not (os.path.exists(register_path))):
        os.mknod(register_path)
        return False

    with open(register_path, "r") as file:
	    data = file.read()
	    files_register = data.split('\n')
    
    return files_register