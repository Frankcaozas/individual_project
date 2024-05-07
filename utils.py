import pandas as pd
import numpy as np

Infinite = float('inf')


def read_data(file_path):
    df = pd.read_excel(file_path)
    data_array = df.values
    return data_array[0:50].T


def material_to_distance(material):
    l = len(material)
    dis_material = [np.zeros(l) for _ in range(l)]
    for i in range(l):
        for j in range(l):
            if (material[i] == material[j]):
                dis_material[i][j] = 0
            else:
                dis_material[i][j] = 1

    for i in range(l):
        for j in range(l):
            if (j == i):
                dis_material[i][j] = Infinite
    return dis_material


def priority_to_distance(priority):
    l = len(priority)
    dis_priority = [np.zeros(l) for _ in range(l)]
    for i in range(l):
        p1 = priority[i]
        for j in range(l):
            p2 = priority[j]
            if (priority[i] == priority[j]):
                dis_priority[i][j] = 0
            elif priority[i] < priority[j]:
                dis_priority[i][j] = 1
            elif p1 > p2 and p1 > p2 + 1:
                dis_priority[i][j] = 2
            else:
                dis_priority[i][j] = 3
    for i in range(l):
        for j in range(l):
            if (j == i):
                dis_priority[i][j] = Infinite

    return dis_priority


def calculate_processing_time(part_specification):
    l = len(part_specification)
    processing_time = np.zeros(l)
    for i in range(l):
        size_str = part_specification[i].split('*')

        if (size_str[-1] == ''):
            size_str = size_str[0:-1]
        processing_time[i] = (
            float(size_str[0]) * float(size_str[1])) * 2 * float(size_str[2]) / 6000 / 0.17
    return processing_time


def obj(material, priority, quntity):
    obj = 0
    l = len(material)
    for i in range(1, l):
        if (quntity[i] > quntity[i-1]):
            obj += 0.3
        if (material[i] != material[i-1]):
            obj += 0.5
        if(priority[i] > priority[i-1]):
            obj += 0.2
    return obj