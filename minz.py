from gurobipy import Model, GRB, quicksum
# from utils import read_data, material_to_distance, calculate_processing_time, priority_to_distance, obj

def test(m, p, q):
  model = Model("zmin_constraints")

  quantity = q  
  material = m
  priority = p
  print(q, m, p)
  num_tasks = len(quantity)  


  delta_qty = model.addVars(num_tasks, vtype=GRB.BINARY, name="delta_qty")
  delta_mat = model.addVars(num_tasks, vtype=GRB.BINARY, name="delta_mat")
  delta_pri = model.addVars(num_tasks, vtype=GRB.BINARY, name="delta_pri")
  zmin = model.addVar(name="zmin")

  
  M = 1000
  for i in range(1, num_tasks):
      model.addConstr(delta_qty[i] >= (quantity[i] - quantity[i-1]) / M, "delta_qty_constr_{}".format(i))

  for i in range(1, num_tasks):
      model.addConstr(delta_mat[i] >= (material[i] - material[i-1]) / M, "delta_mat_constr_{}".format(i))

  for i in range(1, num_tasks):
      model.addConstr(delta_pri[i] >= (priority[i] - priority[i-1]) / M, "delta_pri_constr_{}".format(i))


  model.addConstr(zmin == quicksum(delta_qty[i] + delta_mat[i] + delta_pri[i] for i in range(num_tasks)), "zmin_calc")

  # 设置目标函数
  model.setObjective(zmin, GRB.MINIMIZE)


  model.optimize()


  if model.status == GRB.OPTIMAL:
      print("Optimal Zmin:", zmin.X)
      for i in range(num_tasks):
          print(f"Task {i}: Delta_Quantity={delta_qty[i].X}, Delta_Material={delta_mat[i].X}, Delta_Priority={delta_pri[i].X}")

