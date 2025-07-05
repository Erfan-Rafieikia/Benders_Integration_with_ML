from dataclasses import dataclass
import numpy as np

# ============================
# UFL Problem Data Structure
# ============================
@dataclass # python data class for storing all the data needed to describe an Uncapacitated Facility Location (UFL) problem instance
class UFLData:
    C: np.ndarray  # Customer indices (i in \mathcal{C})
    F: np.ndarray  # Facility indices (j in \mathcal{F})
    S: np.ndarray  # Scenario indices (each customer as subproblem)
    d: np.ndarray  # Customer demands (d_i)
    u: np.ndarray  # Facility capacities (u_j)
    f: np.ndarray  # Facility fixed costs (f_j)
    c: np.ndarray  # Cost matrix (c_{ij})
    total_demand: float
    total_capacity: float

# ============================
# Placeholder Data Structures
# ============================
@dataclass
class HUBData:
    pass

@dataclass
class CMNDData:
    pass

@dataclass
class MCFLData:
    pass

@dataclass
class SSLPData:
    pass

# ============================
# UFL Reader Implementation
# ============================
def read_ufl_data(file_path):
    with open(file_path, "r") as file:
        n_facilities, n_customers = map(int, file.readline().split()) #split the first line to get number of facilities and customers

        u = []  # Facility capacities
        f = []  # Facility fixed costs
        for _ in range(n_facilities): #Goes to second line and reads the next n_facilities lines
            capacity, cost = map(float, file.readline().split())
            u.append(capacity)
            f.append(cost)

        d = []   # Customer demands
        c = []   # Shipment costs. element c[i][j] is the cost of serving customer i from facility j
        for _ in range(n_customers):
            demand_line = file.readline()
            cost_line = file.readline()

            demand = float(demand_line.strip())
            costs = list(map(float, cost_line.strip().split()))

            if len(costs) != n_facilities:
                raise ValueError(f"Expected {n_facilities} costs, got {len(costs)}")

            d.append(demand)
            c.append(costs)

    C = np.arange(n_customers)     # Customer indices . This gives [0, 1, ..., n_customers-1]
    F = np.arange(n_facilities)    # Facility indices. This gives [0, 1, ..., n_facilities-1]. For example, if there are 3 facilities, F will be [0, 1, 2]
    S = C.copy()                   # Each customer is its own scenario . This gives [0, 1, ..., n_customers-1]. Therefore, starting from 0, dpesn't cause concern when defining subproblem for that sccenario

    d = np.array(d,dtype=float) #Shape (|C|,). d[i] is the demand of customer i. For example, if there are 5 customers, d will be an array of length 5 where d[i] is the demand of customer i.
    u = np.array(u,dtype=float) #Shape (|F|,). u[j] is the capacity of facility j. For example, if there are 3 facilities, u will be an array of length 3 where u[j] is the capacity of facility j.
    f = np.array(f,dtype=float) #Shape (|F|,). f[j] is the fixed cost of facility j. For example, if there are 3 facilities, f will be an array of length 3 where f[j] is the fixed cost of facility j.
    #c = np.array(c)  # shape (|C|, |F|). element c[i][j] is the cost of serving customer i from facility j. For example, if there are 5 customers and 3 facilities, c will be a 5x3 matrix where c[i][j] is the cost of serving customer i from facility j.
    
    # Validate cost matrix shape
    for i, row in enumerate(c): # shape (|C|, |F|). element c[i][j] is the cost of serving customer i from facility j. For example, if there are 5 customers and 3 facilities, c will be a 5x3 matrix where c[i][j] is the cost of serving customer i from facility j.
    # Validate cost matrix shape
        if len(row) != n_facilities:
            raise ValueError(f"Row {i} in cost matrix has {len(row)} entries, expected {n_facilities}")
    c = np.array(c, dtype=float)  # Convert to numpy array. Shape (|C|, |F|). element c[i][j] is the cost of serving customer i from facility j. For example, if there are 5 customers and 3 facilities, c will be a 5x3 matrix where c[i][j] is the cost of serving customer i from facility j.


    total_demand = np.sum(d) # Total demand across all customers
    total_capacity = np.sum(u) # Total capacity across all facilities

    print("[UFL] Facilities (F):", F)
    print("[UFL] Customers (C):", C)
    print("[UFL] Demands (d):", d)
    print("[UFL] Capacities (u):", u)
    print("[UFL] Fixed Costs (f):", f)
    print("[UFL] Cost Matrix (c):\n", c)
    print(f"[UFL] Total Demand: {total_demand}, Total Capacity: {total_capacity}")

    return UFLData(
        C=C,
        F=F,
        S=S,
        d=d,
        u=u,
        f=f,
        c=c,
        total_demand=total_demand,
        total_capacity=total_capacity
    ) #creates an instance of UFLData with the read data which is an instance of problem class UFL

# ============================
# Dispatch Reader by Problem Type
# ============================
def read_hub_data(file_path):
    raise NotImplementedError("HUB problem class support not implemented yet.")

def read_cmnd_data(file_path):
    raise NotImplementedError("CMND problem class support not implemented yet.")

def read_mcfl_data(file_path):
    raise NotImplementedError("MCFL problem class support not implemented yet.")

def read_sslp_data(file_path):
    raise NotImplementedError("SSLP problem class support not implemented yet.")

def read_problem_data(problem_type, file_path):
    if problem_type.upper() == "UFL":
        return read_ufl_data(file_path) #return the return of the read_ufl_data function which is an instance of UFLData class sorting all the data needed to describe an Uncapacitated Facility Location (UFL) problem instance.
    elif problem_type.upper() == "HUB":
        return read_hub_data(file_path)
    elif problem_type.upper() == "CMND":
        return read_cmnd_data(file_path)
    elif problem_type.upper() == "MCFL":
        return read_mcfl_data(file_path)
    elif problem_type.upper() == "SSLP":
        return read_sslp_data(file_path)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
