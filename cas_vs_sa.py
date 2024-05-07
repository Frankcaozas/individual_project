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
        
        # Accept the new schedule based on the probability
        if random.random() < acceptance_probability:
            current_schedule = new_schedule
            current_energy = new_energy
            
        # Cooling
        temperature *= cooling_rate
    
    return current_schedule, current_energy

# Chaotic Simulated Annealing with a simple chaos factor
def chaotic_simulated_annealing(initial_schedule, job_times, initial_temp, cooling_rate, temp_threshold, machine_count):
    # Use the logistic map for chaos
    logistic_map = lambda x: 4 * x * (1 - x)  # Chaos function
    x = random.random()  # Initial value for logistic map
    
    current_schedule = initial_schedule.copy()
    current_energy = total_completion_time(current_schedule, job_times, machine_count)
    temperature = initial_temp
    
    while temperature > temp_threshold:
        # Generate a neighbor with chaos influence
        new_schedule = current_schedule.copy()
        idx1, idx2 = random.sample(range(len(new_schedule)), 2)
        new_schedule[idx1], new_schedule[idx2] = new_schedule[idx2], new_schedule[idx1]
        
        new_energy = total_completion_time(new_schedule, job_times, machine_count)
        
        # Acceptance probability with chaos
        if new_energy < current_energy:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp((current_energy - new_energy) / (temperature * logistic_map(x)))
        
        # Accept the new schedule based on the probability
        if random.random() < acceptance_probability:
            current_schedule = new_schedule
            current_energy = new_energy
            
        # Cooling with chaos
        temperature *= cooling_rate * logistic_map(x)
        x = logistic_map(x)  # Update chaos factor
    
    return current_schedule, current_energy

# Generate initial job times and initial schedule
job_count = 30
machine_count = 5
job_times = [random.randint(1, 100) for _ in range(job_count)]
initial_schedule = list(range(job_count))

# Parameters for SA and CSA
initial_temp = 10000
cooling_rate = 0.99
temp_threshold = 0.01
num_iterations = 100  # Number of test iterations

# Conduct the tests
sa_results = []
csa_results = []

for _ in range(num_iterations):
    # Test traditional SA
    sa_schedule, sa_energy = simulated_annealing(
        initial_schedule, job_times, initial_temp, cooling_rate, temp_threshold, machine_count
    )
    sa_results.append(sa_energy)

    # Test CSA
    csa_schedule, csa_energy = chaotic_simulated_annealing(
        initial_schedule, job_times, initial_temp, cooling_rate, temp_threshold, machine_count
    )
    csa_results.append(csa_energy)

# Plot results to visualize the difference in performance
# plt.plot(sa_results, label='Simulated Annealing')
plt.plot(csa_results, label='Chaotic Simulated Annealing')
plt.xlabel('Iteration') 
plt.ylabel('Total Completion Time')
# plt.title('SA vs CSA Performance')
plt.legend()
plt.show()
