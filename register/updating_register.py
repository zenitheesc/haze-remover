def new_coordinates(fileNames, register_path="./coordinates.txt"):
    with open(register_path, "a") as register:
    ## Move read cursor to the start of file.
    #register.seek(0)
    ## If file is not empty then append '\n'
    #data = register.read(100)
    #if len(data) > 0 :
    #    register.write("\n")
        for file in fileNames:
            register.write(file + "\n")