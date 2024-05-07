import numpy as np
from random import randrange
from utils import read_data, material_to_distance, calculate_processing_time, priority_to_distance, obj
import matplotlib.pyplot as plt
import pandas as pd

Infinite = float('inf')

def randi(x, y):
    return randrange(x, y)

def read_data(file_path):
    df = pd.read_excel(file_path)
    data_array = df.values
    return data_array[0:30].T

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

def calculate_cmax_and_schedule(solution):
    machine_times = np.zeros(num_machines)
    machine_schedules = {i: [] for i in range(num_machines)}  
    for task, machine in enumerate(solution):
        machine_times[machine] += task_processing_times[task]
        machine_schedules[machine].append(task) 
    cmax = max(machine_times)
    return cmax, machine_schedules

def logistic_map(x, r=4):
    return r * x * (1 - x)

# 生成新解


def generate_new_solution(current_solution, chaotic_var):
    new_solution = current_solution.copy()
    for i in range(num_tasks):
        # use chaotic var to change the probability to get a solution
        if np.random.rand() < chaotic_var:  
            new_solution[i] = np.random.randint(0, num_machines)
    return new_solution


def simulated_annealing(initial_temp, cooling_rate, max_iterations):
    current_solution = np.random.randint(0, num_machines, size=num_tasks)
    current_cmax, _ = calculate_cmax_and_schedule(current_solution)
    best_solution = current_solution.copy()
    best_cmax = current_cmax
    chaotic_var = np.random.rand()

    for i in range(max_iterations):
        temp = initial_temp * (cooling_rate ** i)
        chaotic_var = logistic_map(chaotic_var)
        new_solution = generate_new_solution(current_solution, chaotic_var)
        new_cmax, _ = calculate_cmax_and_schedule(new_solution)
        energy_diff = new_cmax - current_cmax

        if energy_diff < 0 or np.exp(-energy_diff / temp) > np.random.rand():
            current_solution = new_solution.copy()
            current_cmax = new_cmax

            if new_cmax < best_cmax:
                best_solution = new_solution.copy()
                best_cmax = new_cmax

    best_cmax, best_schedule = calculate_cmax_and_schedule(best_solution)
    return best_solution, best_cmax, best_schedule


def local_greedy_search(distance):
    # Fill the distance matrix
    l = len(distance)

    # Apply the local greedy search algorithm as per the pseudocode
    x1, x2 = np.unravel_index(np.argmin(distance, axis=None), distance.shape)

    sorted = [x1, x2]

    min_a = None
    min_b = None
    used = [False for _ in range(l)]

    used[x1] = True
    used[x2] = True

    # Iteratively find the minimal distance pair and update the sorted list
    for i in range(l-2):
        min_disa = 10
        min_disb = 10
        for j in range(l):
            # print(used[j], j, used[j] or j == sorted[0])
            if used[j] or j == sorted[0]:
                continue

            if distance[j][sorted[0]] < min_disa:
                min_disa = distance[j, sorted[0]]
                min_a = j

        for j in range(l):
            if used[j] or sorted[-1] == j:
                continue

            if distance[sorted[-1]][j] < min_disb:
                min_disb = distance[sorted[-1]][j]
                min_b = j
        if min_disa < min_disb:
            sorted.insert(0, min_a)
            used[min_a] = True
        else:
            sorted.append(min_b)
            used[min_b] = True
    return sorted

def random_neighborhood_search(sorted):
    l = len(sorted)
    re_sorted = np.zeros(l)
    maxCount = 1000
    count = 0
    for i in range(maxCount):
        g = randi(0, 100)
        a = randi(0, l)
        if g < 50:
            s1 = 10
            left = max(0, a-s1)
            right = min(l, a + s1)
            b = a
            while (a == b):
                b = randi(left, right)
            re_sorted = sorted
            re_sorted[a], re_sorted[b] = re_sorted[b], re_sorted[a]
        else:
            s2 = 10
            left = max(0, a-s2)
            right = min(l, a+s2)
            b = a
            while (a == b):
                b = randi(left, right)
            # reomove the part from index a to b then insert them to the end
            re_sorted = sorted
            removed = re_sorted[a:b]
            re_sorted = re_sorted[:a] + re_sorted[b:]
            re_sorted += removed

        # calculate minZ of sorted and resorted obj1 obj2
        m1 = [material[j] for j in sorted]
        p1 = [priority[j] for j in sorted]
        q1 = [processing_quantity[j] for j in sorted]
        m2 = [material[j] for j in re_sorted]
        p2 = [priority[j] for j in re_sorted]
        q2 = [processing_quantity[j] for j in re_sorted]
        obj1 = obj(m1, p1, q1)
        obj2 = obj(m2, p2, q2)
        if (obj2 < obj1):
            sorted = re_sorted,
            count = 0
        else:
            count = count + 1
        return sorted

def sortByQuntity(list):
    def swap(arr, a, b):
        arr[a], arr[b] = arr[b], arr[a]
    new_list = list[:]
    print("enter quntity sort")
    print(new_list, list)
    length = len(list)
    for i in range(length-1, 0, -1):
        # print(i, processing_quantity[i], processing_quantity[i-1],material[i],material[i-1],processing_quantity[i] > processing_quantity[i-1] and material[i] == material[i-1])
        j = i
        cur = new_list[j]
        pre = new_list[j-1]
        while(material[cur] == material[pre]):
            
            print('enter while')
            print(processing_quantity[cur], processing_quantity[pre], processing_quantity[cur] > processing_quantity[pre])
            if (processing_quantity[cur] > processing_quantity[pre] ):
                # print(f"idx:{i} m:{material[i]} {material[i-1]} q:{processing_quantity[i]} {processing_quantity[i-1]}")
                swap(new_list, j, j-1)
                print(f'swap {[processing_quantity[k] for k in new_list], j, j-1}')
            j -= 1
            cur = new_list[j]
            pre = new_list[j-1]
    print('\n')
    return new_list


def single_machine_optimization(machine_schedules):
    op_machine_schedules = {i: [] for i in range(num_machines)}
    for i, ids in machine_schedules.items():
        length = len(ids)
        # print(f" machie{i} {ids} ")
        m = [material[j] for j in ids]
        p = [priority[j] for j in ids]
        # q = [processing_quantity[j] for j in ids]

        dis_material = material_to_distance(m)
        dis_priority = priority_to_distance(p)

        distance = np.zeros((length, length))
        for k in range(length):
            for l in range(length):
                distance[k][l] = dis_material[k][l] + dis_priority[k][l]
        # op_schedule = local_greedy_search(distance)

        op_schedule = [ids[idx] for idx in local_greedy_search(distance)]
        # print(f"before {op_schedule}")
        print(f"machine {i} PHASE1")
        for task in op_schedule:
            # print(f"  Task {task} (Completion Time: {completion_time})")
            print(
                f"  Task {task} Priority {priority[task]} material {material[task]} quntity {processing_quantity[task]}")
        print('\n')

        sb[i] = op_schedule
        # op_schedule = random_neighborhood_search(op_schedule)

        print(f"machine {i} PHASE2")
        for task in op_schedule:
            # print(f"  Task {task} (Completion Time: {completion_time})")
            print(
                f"  Task {task} Priority {priority[task]} material {material[task]} quntity {processing_quantity[task]}")
        print('\n')
        op_schedule = sortByQuntity(op_schedule)

        print(f"machine {i} PHASE3")
        for task in op_schedule:
            # print(f"  Task {task} (Completion Time: {completion_time})")
            print(
                f"  Task {task} Priority {priority[task]} material {material[task]} quntity {processing_quantity[task]}")
        op_machine_schedules[i] = op_schedule

    return op_machine_schedules


def plot(schedle1, schedule2):
    material_color_map = {}

    def get_unique_color(material, color_map):
        if material not in color_map:
            color_map[material] = plt.cm.tab20(len(color_map) % 20)
        return color_map[material]

    def p(machine_schedules):
        fig, gnt = plt.subplots(figsize=(10, 5))

        # Setting labels for x-axis and y-axis
        gnt.set_xlabel('Time')
        gnt.set_ylabel('Machine ID')

        # Setting y-ticks to correspond to machines
        machine_ids = list(machine_schedules.keys())
        gnt.set_yticks([10 * (i + 1) for i in machine_ids])
        gnt.set_yticklabels(['Machine {}'.format(i) for i in machine_ids])

        # Setting graph attribute
        gnt.grid(True)

        # The start time for the first task on each machine is assumed to be zero for simplicity
        start_times = {machine: 0 for machine in machine_ids}

        for machine, tasks in machine_schedules.items():
            for task_id in tasks:
                # Get the task details
                task_priority = priority[task_id]
                task_material = material[task_id]
                task_quantity = processing_quantity[task_id]
                task_processing_time = processing_time[task_id]

                # Get the unique color for the material
                task_color = get_unique_color(
                    task_material, material_color_map)

                # Draw the task bar
                gnt.broken_barh([(start_times[machine], task_processing_time)],
                                (10 * (machine + 1) - 2.5, 5), facecolors=(task_color), edgecolor='black', linewidth=1)

                # Annotate the task bar with task details
                gnt.text(start_times[machine] + task_processing_time / 2, 10 * (machine + 1),
                         f'{task_id}\n{task_priority}\n{task_quantity}',
                         ha='center', va='center', fontsize=6, color='black')

                # Update the start time for the next task
                start_times[machine] += task_processing_time
    plt.figure(1)
    p(schedle1)
    # plt.title('')

    plt.figure(2)
    p(schedule2)
    # plt.title('2 Window Plot')
    plt.show()


file_path = './data.xlsx'
Infinite = float('inf')


data_array = read_data(file_path)
data_array = data_array
l = data_array.shape[1]
task_number, priority, part_code, processing_quantity, part_specification, _,  material, machine = data_array
processing_time = calculate_processing_time(part_specification)


num_tasks = l
num_machines = 2
task_processing_times = processing_time
initial_temp = 100
cooling_rate = 0.95
max_iterations = 1000


best_solution, best_cmax, best_schedule = simulated_annealing(
    initial_temp, cooling_rate, max_iterations)

machine_shcedules = single_machine_optimization(best_schedule)
# print(machine_shcedules)
print("Best Solution:", best_solution)
print("Best Cmax:", best_cmax)
for i in range(num_machines):
    print(f"Machine {i}:")
    for task in machine_shcedules[i]:
        # print(f"  Task {task} (Completion Time: {completion_time})")
        print(
            f"  Task {task} Priority {priority[task]} material {material[task]} quntity {processing_quantity[task]}")
print("\n")


plot(best_schedule, machine_shcedules)


l = sortByQuntity([2, 1, 5, 13,4, 7,21,14,16,18,19,20,23,27])
print([processing_quantity[i] for i in l])
for task in l:
        # print(f"  Task {task} (Completion Time: {completion_time})")
        print(
            f"  Task {task} Priority {priority[task]} material {material[task]} quntity {processing_quantity[task]}")

