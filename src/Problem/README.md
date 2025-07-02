# Uncapacitated Facility Location (UFL)

This repository provides a decomposition-based solution framework (Benders Decomposition) for solving the **Uncapacitated Facility Location (UFL)** problem. In this version, facilities have **no capacity limits**, and each customer must be assigned to exactly one **open facility**.

## Dataset Structure

The benchmark instances follow this format:

1. **First line**:

   ```
   <number_of_facilities> <number_of_customers>
   ```

2. **Next n lines** (for each facility):

   ```
   <ignored_value> <opening_cost_j>
   ```

   > Note: The first value (capacity) is ignored in this uncapacitated version.

3. **Following m lines** (for each customer):

   ```
   <demand_i> <cost_i1> <cost_i2> ... <cost_in>
   ```

## Problem Description

Given:

* A set of **facilities** $j \in \mathcal{F}$,
* A set of **customers** $i \in \mathcal{C}$,
* Each customer $i$ has a **demand** $d_i$,
* Each facility $j$ has a **fixed cost** $f_j$,
* The cost to connect customer $i$ to facility $j$ is $c_{ij}$,

the goal is to:

* Decide which facilities to **open**,
* Assign each customer’s demand to an **open facility**,
* Ensure all customer demand is met,
* **Minimize** total cost (facility opening + connection costs).

## Mathematical Formulations

### 1. Full UFL Formulation (Primal)

Let:
* $y_j \in \{0,1\}$: 1 if facility $j$ is opened,
* $x_{ij} \in [0,1]$: fraction of demand of customer $i$ assigned to facility $j$,

Then the formulation is:


$$
\begin{aligned}
\min_{x, y} \quad & \sum_{j \in \mathcal{F}} f_j y_j + \sum_{i \in \mathcal{C}} \sum_{j \in \mathcal{F}} c_{ij} d_i x_{ij} \\
\text{s.t.} \quad & \sum_{j \in \mathcal{F}} x_{ij} = 1, && \forall i \in \mathcal{C} \\
& x_{ij} \leq y_j, && \forall i, j \\
& x_{ij} \in [0, 1], && \forall i, j \\
& y_j \in \{0, 1\}, && \forall j \in \mathcal{F}
\end{aligned}
$$

### 2. Master Problem

Let $ \theta_i $ be the recourse cost from the subproblem for customer \( i \). The master problem is:

$$\begin{aligned}
\min_{y, \theta} \quad & \sum_{j \in \mathcal{F}} f_j y_j + \sum_{i \in \mathcal{C}} \theta_i \\
\text{s.t.} \quad & \text{Benders cuts (added iteratively)} \\
& y_j \in \{0,1\}, \quad \forall j \in \mathcal{F} \\
& \theta_i \geq 0, \quad \forall i \in \mathcal{C}
\end{aligned}$$

### 3. Subproblem for Customer \( i \)

Given fixed values $ \bar{y}_j $, the subproblem for customer \( i \) assigns its demand to open facilities:

$$
\begin{aligned}
Q_i(\bar{y}) := \min_{x_{ij}} \quad & \sum_{j \in \mathcal{F}} c_{ij} d_i x_{ij} \\
\text{s.t.} \quad & \sum_{j \in \mathcal{F}} x_{ij} = 1 \\
& x_{ij} \leq \bar{y}_j, && \forall j \in \mathcal{F} \\
& x_{ij} \geq 0, && \forall j \in \mathcal{F}
\end{aligned}
$$

### 4. Dual of the Subproblem for Customer \( i \)

Let:
- \( \lambda_i \): dual for \( \sum_j x_{ij} = 1 \),
- \( \mu_{ij} \geq 0 \): dual for \( x_{ij} \leq \bar{y}_j \),

The dual becomes:

$$
\begin{aligned}
\max_{\lambda_i, \mu_{ij}} \quad & \lambda_i - \sum_{j \in \mathcal{F}} \mu_{ij} \bar{y}_j \\
\text{s.t.} \quad & \lambda_i - \mu_{ij} \leq c_{ij} d_i, \quad \forall j \in \mathcal{F} \\
& \mu_{ij} \geq 0, \quad \forall j \in \mathcal{F}
\end{aligned}
$$

### 5. Benders Optimality Cut for Customer \( i \)

From the dual solution for customer \( i \), the optimality cut is:

$$
\theta_i \geq \lambda_i^* - \sum_{j \in \mathcal{F}} \mu_{ij}^* y_j
$$

This cut is added to the master problem iteratively.

