from dataclasses import dataclass
import numpy as np

# ============================
# UFL Problem Data Structure
# ============================
@dataclass
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
        n_facilities, n_customers = map(int, file.readline().split())

        u = []  # Facility capacities
        f = []  # Facility fixed costs
        for _ in range(n_facilities):
            capacity, cost = map(float, file.readline().split())
            u.append(capacity)
            f.append(cost)

        d = []   # Customer demands
        c = []   # Shipment costs
        for _ in range(n_customers):
            row = list(map(float, file.readline().split()))
            d.append(row[0])
            c.append(row[1:])

    C = np.arange(n_customers)     # Customer indices
    F = np.arange(n_facilities)    # Facility indices
    S = C.copy()                   # Each customer is its own scenario

    d = np.array(d)
    u = np.array(u)
    f = np.array(f)
    c = np.array(c)  # shape (|C|, |F|)

    total_demand = np.sum(d)
    total_capacity = np.sum(u)

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
    )

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
        return read_ufl_data(file_path)
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
