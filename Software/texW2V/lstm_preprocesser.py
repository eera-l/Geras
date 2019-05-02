import os
import pandas as pd
import numpy as np

pathControl = '/home/federica/Documents/Thesis/Programs/TextParser/control/control_no_functional_clean'
pathDementia = '/home/federica/Documents/Thesis/Programs/TextParser/dementia/dementia_no_functional_clean'

filesC = []
filesD = []
rows_control = []
rows_dementia = []


def read_directory(files, path):
    for r, d, f in os.walk(path):
        for file in f:
            if '.cha' in file:
                files.append(os.path.join(r, file))


def read_files(files, start_range, end_range, control):
    global lines, rows_control, rows_dementia
    lines = []
    for i in range(start_range, end_range):
        file = files[i]
        file = open(file, "r")
        lines = file.read()
        lines = lines.split("\n")[:-1]
        one_line = ""
        for line in lines:
            if line is "\n":
                continue
            words_line = line.split(" ")[:-1]
            for word in words_line:
                if word is "":
                    continue
                one_line += word + " "
        if control:
            temp_array = [one_line, 'N']
            rows_control.append(temp_array)
        else:
            temp_array = [one_line, 'Y']
            rows_dementia.append(temp_array)


def write_to_file(title):
    global rows, rows_control, rows_dementia
    rows = []
    rows = rows_control + rows_dementia
    rows_control = []
    rows_dementia = []
    df = pd.DataFrame(np.array(rows), columns=['text', 'dementia'])
    df.to_csv(title, index=False)


read_directory(filesC, pathControl)
read_directory(filesD, pathDementia)
read_files(filesC, 0, 206, True)
read_files(filesD, 0, 263, False)
write_to_file('train_set_lstm')
read_files(filesC, 206, len(filesC), True)
read_files(filesD, 263, len(filesD), False)
write_to_file('test_set_lstm')

