import sys
import numpy as np
import mosek.fusion as mf
from wdro_inner.solver import InnerMaxResult
from wdro_inner.losses import PointwiseMaxLoss, RadialLossComponent

class MosekQuadraticSolver:
    """
    MOSEK Fusion solver for W-DRO inner maximization with Pointwise Maximum of Quadratic Losses.
    Acts as a drop-in benchmark replacement for InnerSolverl2.
    """
    
    def __init__(self, epsilon: float, verbose: bool = False):
        self.epsilon = epsilon
        self.verbose = verbose

    def solve(self, z_empirical: np.ndarray, loss: PointwiseMaxLoss) -> InnerMaxResult:
        N, d = z_empirical.shape
        K = loss.K
        B = self.epsilon * N  # Total budget across all N samples

        # 1. Extract matrices and precompute c_{i,k} = A_k z_i + b_k
        A_list = []
        intercepts = []
        c_ik_list = []      # Will store an (N, m_k) array for each component
        c_norm_sq_list = [] # Will store an (N,) array for each component

        for k, comp in enumerate(loss.components):
            if not isinstance(comp, RadialLossComponent):
                raise ValueError("MOSEK solver requires RadialLossComponents.")
            
            A_list.append(comp.A)
            intercepts.append(comp.intercept)

            # Vectorized precomputation of c_{i,k} and its norm
            # comp.A is (m_k, d) and z_empirical is (N, d)
            c_k = z_empirical @ comp.A.T + comp.b_vec.reshape(1, -1)
            c_ik_list.append(c_k)
            c_norm_sq_list.append(np.sum(c_k ** 2, axis=1))

        # 2. Build MOSEK Model
        with mf.Model("WDRO_Inner_Max_Quad_Mosek") as M:
            if self.verbose:
                M.setLogHandler(sys.stdout)
            
            # --- Variables ---
            alpha = M.variable("alpha", [N, K], mf.Domain.inRange(0.0, 1.0))
            q = M.variable("q", [N, K, d], mf.Domain.unbounded())
            norm_q = M.variable("norm_q", [N, K], mf.Domain.greaterThan(0.0))
            t = M.variable("t", [N, K], mf.Domain.greaterThan(0.0))

            # --- Constraints ---
            # 1. Convex combination: sum_k alpha_{i,k} = 1 (Vectorized over N)
            M.constraint("sum_alpha", mf.Expr.sum(alpha, 1), mf.Domain.equalsTo(1.0))

            # 2. Global Budget Constraint
            budget_constr = M.constraint("global_budget", mf.Expr.sum(norm_q), mf.Domain.lessThan(B))

            # 3. Norm definition: norm_q_{i,k} >= ||q_{i,k}||_2
            # Stacking norm_q and q to form N*K standard quadratic cones
            norm_q_flat = mf.Expr.reshape(norm_q, [N * K, 1])
            q_flat = mf.Expr.reshape(q, [N * K, d])
            M.constraint("norm_def", mf.Expr.hstack(norm_q_flat, q_flat), mf.Domain.inQCone())

            obj_terms = []

            # 4 & 5. y definition & RSOC Perspective Constraint (per component)
            for k in range(K):
                m_k = A_list[k].shape[0]
                
                # Extract q for component k: shape [N, d]
                q_k = mf.Expr.reshape(q.slice([0, k, 0], [N, k+1, d]), [N, d])
                
                # Define y_{i,k} = A_k q_{i,k}. Matrix multiplication handles all N at once
                A_k_T = mf.Matrix.dense(A_list[k].T)
                y_k = mf.Expr.mul(q_k, A_k_T)  # shape: [N, m_k]

                # Extract column vectors for t and alpha
                t_k_col = mf.Expr.reshape(t.slice([0, k], [N, k+1]), [N, 1])
                alpha_k_col = mf.Expr.reshape(alpha.slice([0, k], [N, k+1]), [N, 1])

                # MOSEK Rotated Quadratic Cone definition: 2 * x_0 * x_1 >= sum(x_i^2)
                # To enforce t * alpha >= ||y||_2^2, we map: x_0 = 0.5 * t, x_1 = alpha
                half_t_k_col = mf.Expr.mul(0.5, t_k_col)
                rsoc_expr = mf.Expr.hstack(half_t_k_col, alpha_k_col, y_k)
                
                # Passing a 2D matrix to inRotatedQCone creates a cone for each row independently
                M.constraint(f"rsoc_{k}", rsoc_expr, mf.Domain.inRotatedQCone())

                # --- Objective Construction ---
                # obj = sum [ alpha * (intercept - ||c||^2) + 2 * c^T y - t ]
                alpha_k_flat = mf.Expr.flatten(alpha_k_col)
                coeff_alpha_k = intercepts[k] - c_norm_sq_list[k]
                obj_alpha_k = mf.Expr.dot(coeff_alpha_k, alpha_k_flat)

                c_k_flat = c_ik_list[k].flatten()
                y_k_flat = mf.Expr.flatten(y_k)
                obj_y_k = mf.Expr.mul(2.0, mf.Expr.dot(c_k_flat, y_k_flat))

                t_k_flat = mf.Expr.flatten(t_k_col)
                obj_t_k = mf.Expr.sum(t_k_flat)

                comp_obj = mf.Expr.sub(mf.Expr.add(obj_alpha_k, obj_y_k), obj_t_k)
                obj_terms.append(comp_obj)

            # Maximize the total sum
            M.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.add(obj_terms))
            
            # Set wall-clock time limit
            M.setSolverParam("optimizerMaxTime", 300.0)

            # --- Solve ---
            M.solve()

            sol_sta = M.getPrimalSolutionStatus()
            if sol_sta != mf.SolutionStatus.Optimal:
                raise RuntimeError(f"MOSEK failed to find optimal solution. Status code: {sol_sta}")

            # --- Extract Results ---
            # Convert total objective to empirical average
            obj_val = M.primalObjValue() / N  
            
            # Extract dual variable for budget constraint (magnitude is the optimal lambda)
            lambda_opt = abs(budget_constr.dual()[0]) 
            
            optimal_budgets = np.zeros(N)
            weights_list = []
            support_points_list = []
            active_comps_list = []

            # Retrieve primal variable values and reshape back to expected dimensions
            alpha_val = alpha.level().reshape((N, K))
            q_val = q.level().reshape((N, K, d))
            norm_q_val = norm_q.level().reshape((N, K))

            for i in range(N):
                optimal_budgets[i] = np.sum(norm_q_val[i, :])
                active_k_for_i = []
                
                for k in range(K):
                    a_val = alpha_val[i, k]
                    if a_val > 1e-8:
                        weight = a_val / N
                        point = z_empirical[i] - (q_val[i, k, :] / a_val)
                        
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