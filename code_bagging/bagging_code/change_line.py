
def modify_line_file(num_line, path_file, new_text):

    with open(path_file, 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # Change line, note that you have to add a newline
    data[num_line - 1] = new_text + '\n'

    # And write everything back to file
    with open(path_file, 'w') as file:
        file.writelines( data )  