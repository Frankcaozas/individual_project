# import numpy as np
from gurobipy import Model, GRB, quicksum
import numpy as np
# 假设参数
num_tasks = 10  
num_machines = 3  
task_processing_times = np.random.randint(1, 10, size=num_tasks)  


def calculate_cmax_and_schedule(solution):
    machine_times = np.zeros(num_machines)
    machine_schedules = {i: [] for i in range(num_machines)}  # 记录每台机器上的任务调度情况
    for task, machine in enumerate(solution):
        machine_times[machine] += task_processing_times[task]
        machine_schedules[machine].append((task, machine_times[machine]))  # 记录任务和其完成时间
    cmax = max(machine_times)
    return cmax, machine_schedules

# Logistic映射
def logistic_map(x, r=4):
    return r * x * (1 - x)

# 生成新解
def generate_new_solution(current_solution, chaotic_var):
    new_solution = current_solution.copy()
    for i in range(num_tasks):
        if np.random.rand() < chaotic_var:  # 使用混沌变量作为改变解的概率
            new_solution[i] = np.random.randint(0, num_machines)  # 随机改变任务的机器分配
    return new_solution

# 模拟退火算法
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

# 设置算法参数
initial_temp = 100  
cooling_rate = 0.95  
max_iterations = 1000  

# 运行模拟退火算法
best_solution, best_cmax, best_schedule = simulated_annealing(initial_temp, cooling_rate, max_iterations)

print("Best Solution:", best_solution)
print("Best Cmax:", best_cmax)
for machine, tasks in best_schedule.items():
    print(f"Machine {machine}:")
    for task, completion_time in tasks:
        print(f"  Task {task} (Completion Time: {completion_time} ")




def optimize(num_machines, num_tasks, task_processing_times, material, processing_quntity):
  初始化模型
  model = Model("parallel_machine_scheduling")

  # 决策变量：如果任务j分配给机器i，则x[i,j] = 1
  x = model.addVars(num_machines, num_tasks, vtype=GRB.BINARY, name="x")

  # 每个任务只能分配给一个机器
  for j in range(num_tasks):
      model.addConstr(quicksum(x[i, j] for i in range(num_machines)) == 1, name=f"task_{j}")

  # 计算每台机器上任务的完成时间
  completion_times = model.addVars(num_machines, name="completion_times")
  for i in range(num_machines):
      model.addConstr(completion_times[i] == quicksum(x[i, j] * task_processing_times[j] for j in range(num_tasks)), name=f"completion_time_{i}")

  # 引入辅助变量表示最大完成时间
  cmax = model.addVar(name="cmax")

  zmin = model.addVar(name="zmin")

  # 确保cmax大于等于所有机器上的完成时间
  model.addConstrs((cmax >= completion_times[i] for i in range(num_machines)), name="cmax_constraint")
  model.addConstrs

  # 目标函数：最小化最大完成时间
  model.setObjective(cmax, GRB.MINIMIZE)

  model.setObjective(zmin, GRB.MINIMIZE)

  # 求解模型
  model.optimize()

  # 输出结果
  if model.status == GRB.OPTIMAL:
      print(f"Optimal Cmax: {model.objVal}")
      for i in range(num_machines):
          print(f"Machine {i}:")
          for j in range(num_tasks):
              if x[i, j].X > 0.5:  # 如果任务j分配给机器i
                  print(f"  Task {j}")


  

# from gurobipy import Model, GRB, quicksum

# model = Model("zmin_constraints")

# # 假设变量和参数
# quantity = [5, 3, 6, 2]  # 示例数量数组
# material = [1, 2, 1, 2]  # 示例材料数组
# priority = [1, 3, 2, 4]  # 示例优先级数组
# num_tasks = len(quantity)  # 任务数量

# # 添加决策变量
# delta_qty = model.addVars(num_tasks, vtype=GRB.BINARY, name="delta_qty")
# delta_mat = model.addVars(num_tasks, vtype=GRB.BINARY, name="delta_mat")
# delta_pri = model.addVars(num_tasks, vtype=GRB.BINARY, name="delta_pri")
# zmin = model.addVar(name="zmin")

# # 大M
# M = 3

# # 为 delta_qty 添加约束
# for i in range(1, num_tasks):
#     model.addConstr(delta_qty[i] >= (quantity[i] - quantity[i-1]) / M, "delta_qty_constr_{}".format(i))

# # 为 delta_mat 添加约束
# for i in range(1, num_tasks):
#     model.addConstr(delta_mat[i] >= (material[i] - material[i-1]) / M, "delta_mat_constr_{}".format(i))

# # 为 delta_pri 添加约束
# for i in range(1, num_tasks):
#     model.addConstr(delta_pri[i] >= (priority[i] - priority[i-1]) / M, "delta_pri_constr_{}".format(i))

# # 定义 zmin 的计算
# model.addConstr(zmin == quicksum(delta_qty[i] + delta_mat[i] + delta_pri[i] for i in range(num_tasks)), "zmin_calc")

# # 设置目标函数
# model.setObjective(zmin, GRB.MINIMIZE)

# # 求解模型
# model.optimize()

# # 输出结果
# if model.status == GRB.OPTIMAL:
#     print("Optimal Zmin:", zmin.X)
#     for i in range(num_tasks):
#         print(f"Task {i}: Delta_Quantity={delta_qty[i].X}, Delta_Material={delta_mat[i].X}, Delta_Priority={delta_pri[i].X}")
