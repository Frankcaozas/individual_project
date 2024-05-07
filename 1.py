import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Objective function: total completion time for a schedule of jobs on machines
def total_completion_time(schedule, job_times, machine_count):
    # Calculate the completion time for each machine
    machine_times = [0] * machine_count
    for job in schedule:
        # Find the machine with the minimum current time
        min_machine = np.argmin(machine_times)
        # Add the job time to the selected machine
        machine_times[min_machine] += job_times[job]
    return max(machine_times)  # Completion time is the time of the slowest machine

# Simulated Annealing
def simulated_annealing(initial_schedule, job_times, initial_temp, cooling_rate, temp_threshold, machine_count):
    current_schedule = initial_schedule.copy()
    current_energy = total_completion_time(current_schedule, job_times, machine_count)
    temperature = initial_temp
    energies = []  # Keep track of energies
    
    while temperature > temp_threshold:
        # Generate a neighbor by swapping two jobs
        new_schedule = current_schedule.copy()
        idx1, idx2 = random.sample(range(len(new_schedule)), 2)
        new_schedule[idx1], new_schedule[idx2] = new_schedule[idx2], new_schedule[idx1]
        
        new_energy = total_completion_time(new_schedule, job_times, machine_count)
        
        # Acceptance probability
        if new_energy < current_energy:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp((current_energy - new_energy) / temperature)
      
        if random.random() < acceptance_probability:
            current_schedule = new_schedule
            current_energy = new_energy
            
        energies.append(current_energy)
        
        temperature *= cooling_rate
    
    return energies

def chaotic_simulated_annealing(initial_schedule, job_times, initial_temp, cooling_rate, temp_threshold, machine_count):
    
    logistic_map = lambda x: 3.57 * x * (1 - x)  
    x = random.random()
    current_schedule = initial_schedule.copy()
    current_energy = total_completion_time(current_schedule, job_times, machine_count)
    temperature = initial_temp
    energies = []  
    
    while temperature > temp_threshold:
        new_schedule = current_schedule.copy()
        idx1, idx2 = random.sample(range(len(new_schedule)), 2)
        new_schedule[idx1], new_schedule[idx2] = new_schedule[idx2], new_schedule[idx1]
        
        new_energy = total_completion_time(new_schedule, job_times, machine_count)
        
        if new_energy < current_energy:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp((current_energy - new_energy) / temperature) * logistic_map(x)
        
        if random.random() < acceptance_probability:
            current_schedule = new_schedule
            current_energy = new_energy
            
        energies.append(current_energy)
        temperature *= cooling_rate
        if random.random() < 0.1:
            x = logistic_map(x)
    
    return energies

job_count = 50
machine_count = 20
job_times = [random.randint(1, 100) for _ in range(job_count)]
initial_schedule = list(range(job_count))
initial_temp = 100
cooling_rate = 0.95
temp_threshold = 0.01

sa_energies = simulated_annealing(
    initial_schedule, job_times, initial_temp, cooling_rate, temp_threshold, machine_count
)

csa_energies = chaotic_simulated_annealing(
    initial_schedule, job_times, initial_temp, cooling_rate, temp_threshold, machine_count
)

plt.figure(figsize=(10, 5))
plt.plot(sa_energies, label='SA', linestyle='--', color='blue')
plt.plot(csa_energies, label='CSA', linestyle='--', color='red')
plt.xlabel('Iterations')
plt.ylabel('Total Completion Time')
plt.title('Comparison of SA and CSA Convergence')
plt.legend()
plt.grid(True)
plt.show()

