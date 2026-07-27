import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from wdro_inner.solver import InnerMaxResult
from wdro_inner.losses import PointwiseMaxLoss


class GurobiQuadraticSolver:
    """
    Gurobi solver for W-DRO inner maximization with Pointwise Maximum of Quadratic Losses.
    Acts as a drop-in benchmark replacement for InnerSolverl2.
    """
    
    def __init__(self, epsilon: float, verbose: bool = False):
        self.epsilon = epsilon
        self.verbose = verbose

    def solve(self, z_empirical: np.ndarray, loss: PointwiseMaxLoss) -> InnerMaxResult:
        def check_timeout():
                    if time.perf_counter() - start_time > TIMEOUT:
                        raise TimeoutError("Total wall-clock limit of 600s exceeded during setup or solve.")

        TIMEOUT = 600.0
        start_time = time.perf_counter()

        N, d = z_empirical.shape
        K = loss.K
        B = self.epsilon * N  # Total budget across all N samples

        # 1. Extract matrices and precompute c_{i,k} = A_k z_i + b_k
        A_list = [comp.A for comp in loss.components]
        intercepts = np.array([comp.intercept for comp in loss.components])
        c_ik = np.array([[A_list[k] @ z_empirical[i] + comp.b_vec for k, comp in enumerate(loss.components)] for i in range(N)])
        c_norm_sq = np.array([[np.sum(c_ik[i, k]**2) for k in range(K)] for i in range(N)])

        # 2. Build Gurobi Model
        env = gp.Env(empty=True)
        if not self.verbose:
            env.setParam("OutputFlag", 0)
        env.start()
        
        model = gp.Model("WDRO_Inner_Max_Quad", env=env)
        model.Params.TimeLimit = TIMEOUT
        model.Params.QCPDual = 1   # Enable QCP duals to fetch optimal lambda later

        # --- Variables ---
        alpha = model.addVars(N, K, lb=0.0, ub=1.0, name="alpha")
        q = model.addVars(N, K, d, lb=-GRB.INFINITY, name="q")
        norm_q = model.addVars(N, K, lb=0.0, name="norm_q")
        t = model.addVars(N, K, lb=0.0, name="t")

        # Flattened y variables for batching
        y_dims = [A.shape[0] for A in A_list]
        y_vars = {}
        for k in range(K):
            y_vars[k] = model.addVars(N, y_dims[k], lb=-GRB.INFINITY, name=f"y_k{k}")

        # --- Constraints ---
        # 1. Convex combination: sum_k alpha_{i,k} = 1
        model.addConstrs((gp.quicksum(alpha[i, k] for k in range(K)) == 1.0 for i in range(N)), name="sum_alpha")

        # 2. Norm definition: norm_q_{i,k}^2 >= ||q_{i,k}||_2^2
        model.addConstrs((norm_q[i, k] * norm_q[i, k] >= gp.quicksum(q[i, k, j]*q[i, k, j] for j in range(d)) 
                     for i in range(N) for k in range(K)), name="norm_def")

        # 3. Global Budget Constraint
        budget_constr = model.addConstr(gp.quicksum(norm_q[i, k] for i in range(N) for k in range(K)) <= B, name="global_budget")

        # 4. y definition: y_{i,k} = A_k q_{i,k}
        for k in range(K):
            A_k = A_list[k]
            for i in range(N):
                check_timeout()
                for l in range(y_dims[k]):
                    model.addConstr(y_vars[k][i, l] == gp.quicksum(A_k[l, j] * q[i, k, j] for j in range(d)))
        
        # 5. RSOC Perspective Constraint: t_{i,k} * alpha_{i,k} >= ||y_{i,k}||_2^2
        model.addConstrs((t[i, k] * alpha[i, k] >= gp.quicksum(y_vars[k][i, l]*y_vars[k][i, l] for l in range(y_dims[k]))
                     for i in range(N) for k in range(K)), name="rsoc")
        check_timeout()

        # --- Objective ---
        # maximize: sum_{i,k} [ -alpha_{i,k} ||c_{i,k}||^2 + 2 c_{i,k}^T y_{i,k} - t_{i,k} ]
        obj = gp.quicksum(
            alpha[i, k] * intercepts[k] - alpha[i, k] * c_norm_sq[i, k] 
            + 2 * gp.quicksum(c_ik[i, k][l] * y_vars[k][i, l] for l in range(y_dims[k])) - t[i, k]
            for i in range(N) for k in range(K)
        )
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # --- Solve ---
        model.optimize()
        check_timeout()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed to find optimal solution. Status code: {model.status}")

        # --- Extract Results ---
        # Convert total objective to empirical average
        # obj_val = model.ObjVal / N  
        
        # For maximization, the Pi of a <= constraint is positive.
        lambda_opt = abs(budget_constr.Pi) 
        
        optimal_budgets = np.zeros(N)
        weights_list = []
        support_points_list = []
        active_comps_list = []

        for i in range(N):
            optimal_budgets[i] = sum(norm_q[i, k].X for k in range(K))
            active_k_for_i = []
            
            for k in range(K):
                a_val = alpha[i, k].X
                if a_val > 1e-8:
                    q_val = np.array([q[i, k, j].X for j in range(d)])
                    
                    weight = a_val / N
                    point = z_empirical[i] - (q_val / a_val)
                    
                    weights_list.append(weight)
                    support_points_list.append(point)
                    active_k_for_i.append(k)
            
            active_comps_list.append(active_k_for_i)
        
        exact_plugin_loss = 0.0
        for weight, point in zip(weights_list, support_points_list):
                exact_plugin_loss += weight * loss.evaluate(point)

        return InnerMaxResult(
            worst_case_loss=exact_plugin_loss,
            optimal_lambda=lambda_opt,
            optimal_budgets=optimal_budgets,
            worst_case_distribution=(np.array(weights_list), np.array(support_points_list)),
            active_components=active_comps_list
        )