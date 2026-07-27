import numpy as np
import time
import mosek.fusion as mf
import scipy.linalg as la
from typing import Callable
from inner.solver_l2 import InnerSolverl2, numba_phi
from inner.losses import PointwiseMaxLoss, RadialLossComponent
from inner.compression import compress_greedy
from primal.compression_dual_l2 import prepare_dual_compression_data, compress_distribution_dual_quad
from primal.compression_tangent_l2 import prepare_tangent_compression_data, compress_distribution_tangent_quad


class SaddlePointProbleml2:
    """
    Manages the K-piece convex-concave saddle point problem:
    l(x, z) = max_k { -z^T A_k z + z^T B_k x + x^T C_k x }
    where A_k and C_k are PSD matrices.
    """
    def __init__(self, A_list: list, B_list: list, C_list: list, d: int, R: float):
        self.A = np.array(A_list)
        self.B = np.array(B_list)
        self.C = np.array(C_list)
        self.K = len(A_list)
        self.d = d  # dimension of x and z
        self.R = R

        # Precomputed matrices for the inner loss generation
        self.A_half = []
        self.M = []
        
        for k in range(self.K):
            # 1. Ensure A_k and C_k are perfectly symmetric
            self.A[k] = 0.5 * (self.A[k] + self.A[k].T)
            self.C[k] = 0.5 * (self.C[k] + self.C[k].T)
            
            # 2. Eigendecomposition of A_k
            eigvals, eigvecs = np.linalg.eigh(self.A[k])
            
            # Clip negative eigenvalues to 0 to ensure strict PSD.
            # Add a tiny epsilon (1e-10) to ensure A_k is strictly invertible for the square root.
            eigvals_clipped = np.clip(eigvals, 1e-10, None)
            
            # 3. Compute A_k^{1/2} and A_k^{-1/2}
            A_half_k = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
            A_inv_half_k = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_clipped)) @ eigvecs.T
            
            self.A_half.append(A_half_k)
            
            # 4. Precompute the multiplier M_k = -0.5 * A_k^{-1/2} B_k
            self.M.append(-0.5 * A_inv_half_k @ self.B[k])

        self.A_half = np.array(self.A_half)
        self.M = np.array(self.M)

    def build_inner_loss(self, x: np.ndarray) -> PointwiseMaxLoss:
        """
        Locks in a fixed x to generate the PointwiseMaxLoss for the inner solver.
        Completes the square to map to: intercept - || A z + b_vec ||_2^2
        """
        components = []
        for k in range(self.K):
            # A_inner = A_k^{1/2}
            A_inner = self.A_half[k]
            
            # b_vec = M_k * x
            b_vec = self.M[k] @ x
            
            # intercept = x^T C_k x + || b_vec ||_2^2
            intercept = (x.T @ self.C[k] @ x) + np.sum(b_vec**2)
            
            # Create the component utilizing our exact L2 architecture
            comp = RadialLossComponent(A_inner, b_vec, phi=numba_phi, intercept=intercept)
            components.append(comp)
            
        return PointwiseMaxLoss(components)
    
    def expected_loss(self, x: np.ndarray, weights: np.ndarray, supports: np.ndarray) -> float:
        """
        Computes the expected loss E_{z ~ Q} [l(x, z)] for a given distribution Q.
        Vectorized to evaluate all support points simultaneously for maximum speed.
        """
        if len(weights) == 0:
            return 0.0
            
        M = len(weights)
        vals = np.zeros((M, self.K))
        
        # Evaluate all K components for all M support points at once
        for k in range(self.K):
            # Scalar: x^T C_k x
            c_k_x = x.T @ self.C[k] @ x
            
            # Vector (d,): B_k x
            b_k_x = self.B[k] @ x
            
            # Vector (M,): z^T A_k z for all z in supports
            z_A_z = np.sum(supports @ self.A[k] * supports, axis=1)
            
            # Vector (M,): z^T B_k x for all z in supports
            z_B_x = supports @ b_k_x
            
            # Store the evaluations for component k
            vals[:, k] = -z_A_z + z_B_x + c_k_x
            
        # For each support point, find the maximum loss across the K components
        max_vals = np.max(vals, axis=1)  # Shape: (M,)
        
        # Compute the expectation by taking the dot product with the probability weights
        return float(np.dot(weights, max_vals))

    def expected_gradient_x(self, x: np.ndarray, weights: np.ndarray, supports: np.ndarray) -> np.ndarray:
        """
        Computes gradient of expected loss w.r.t x evaluated at the worst-case distribution Q_t.
        By Danskin's Theorem, the gradient of the max is the gradient of the active component.
        """
        grad = np.zeros_like(x)

        for w, z in zip(weights, supports):
            # 1. Identify the active component k* for this specific support point z
            vals = np.zeros(self.K)
            for k in range(self.K):
                # -z^T A_k z + z^T B_k x + x^T C_k x
                vals[k] = -(z.T @ self.A[k] @ z) + (z.T @ self.B[k] @ x) + (x.T @ self.C[k] @ x)
            k_star = np.argmax(vals)
            
            # 2. Compute gradient of the active component w.r.t x: 2 * C_k x + B_k^T z
            g_k = 2 * (self.C[k_star] @ x) + (self.B[k_star].T @ z)
            
            # 3. Accumulate the weighted expected gradient
            grad += w * g_k
            
        return grad
    
    def best_response_x(self, weights: np.ndarray, supports: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Computes the unconstrained optimal decision x for a fixed distribution Q.
        Uses the native MOSEK Fusion API for maximum compilation and solving efficiency.
        """
        if len(weights) == 0:
            return np.zeros(self.d), 0.0
            
        M = len(weights)
        
        with mf.Model("BestResponseQCQP_Fusion") as M_model:
            # --- Variables ---
            # Define as column vectors [size, 1] for easier matrix multiplication in Fusion
            x = M_model.variable("x", [self.d, 1], mf.Domain.unbounded())
            t = M_model.variable("t", [M, 1], mf.Domain.unbounded())

            # Auxiliary variable to linearize the L1 norm: u_i >= |x_i|
            u = M_model.variable("u", [self.d, 1], mf.Domain.greaterThan(0.0))
            
            # --- Constraints ---
            # 0. L1 Trust Region Constraint: ||x||_1 <= R
            # u >= x
            M_model.constraint("l1_upper", mf.Expr.sub(u, x), mf.Domain.greaterThan(0.0))
            # u >= -x
            M_model.constraint("l1_lower", mf.Expr.add(u, x), mf.Domain.greaterThan(0.0))
            # sum(u) <= R
            M_model.constraint("trust_region", mf.Expr.sum(u), mf.Domain.lessThan(self.R))
            # # 0. Trust Region Constraint: ||x||_2 <= R
            # # Mapped to a quadratic cone: [R, x] \in Q^{d+1}
            # x_flat = mf.Expr.flatten(x)
            # M_model.constraint("trust_region", mf.Expr.vstack(self.R, x_flat), mf.Domain.inQCone())

            for k in range(self.K):
                # 1. Precompute Cholesky factor: U_k^T U_k = C_k
                # Since C_k is strictly PSD, this is highly stable
                U_k = la.cholesky(self.C[k], lower=False)
                U_k_mat = mf.Matrix.dense(U_k)
                
                # 2. Precompute constant matrices
                z_A_z = np.sum(supports @ self.A[k] * supports, axis=1)
                z_A_z_col = mf.Matrix.dense(z_A_z.reshape(M, 1))
                
                Z_B_k = supports @ self.B[k]
                Z_B_k_mat = mf.Matrix.dense(Z_B_k)
                
                # 3. Build the epigraph slack: s_k = t + z_A_z - Z_B_k * x
                # Shape: [M, 1]
                linear_k = mf.Expr.mul(Z_B_k_mat, x)
                t_plus_zAz = mf.Expr.add(t, z_A_z_col)
                s_k = mf.Expr.sub(t_plus_zAz, linear_k)
                
                # 4. Build the quadratic norm vector: v_k = U_k * x
                # Shape: [d, 1]
                Ux = mf.Expr.mul(U_k_mat, x)
                
                # Transpose and repeat Ux across M rows to shape [M, d]
                Ux_T = mf.Expr.transpose(Ux)
                ones_col = mf.Matrix.dense(np.ones((M, 1)))
                V_k = mf.Expr.mul(ones_col, Ux_T)
                
                # 5. Rotated Second-Order Cone Perspective Constraint
                # 2 * (0.5) * s_k >= || V_k ||_2^2
                half_col = mf.Expr.constTerm(np.full((M, 1), 0.5))
                rsoc_expr = mf.Expr.hstack(half_col, s_k, V_k)
                
                M_model.constraint(f"rsoc_{k}", rsoc_expr, mf.Domain.inRotatedQCone())
                
            # --- Objective ---
            # Minimize: weights^T * t
            t_flat = mf.Expr.flatten(t)
            M_model.objective("obj", mf.ObjectiveSense.Minimize, mf.Expr.dot(weights, t_flat))
            
            # --- Solver Parameters ---
            M_model.setSolverParam("intpntCoTolRelGap", 1e-6)
            M_model.setSolverParam("numThreads", 0)
            
            # --- Solve ---
            M_model.solve()
            
            sol_sta = M_model.getPrimalSolutionStatus()
            if sol_sta != mf.SolutionStatus.Optimal:
                raise RuntimeError(f"MOSEK Fusion failed to find optimal x. Status: {sol_sta}")
                
            # --- Extract ---
            x_star = x.level().flatten()
            opt_val = M_model.primalObjValue()
            
        return x_star, opt_val


def update_support_history(S_history: list, result, N: int):
    """
    Separates the flat array of worst-case support points by slicing it according 
    to the number of active components recorded for each empirical sample i.
    """
    _, final_supports = result.worst_case_distribution
    active_comps = result.active_components  # List of N lists
    
    current_idx = 0
    
    for i in range(N):
        # The number of points generated for sample i is exactly the length 
        # of its corresponding active components sublist
        num_points = len(active_comps[i])

        if num_points == 0:
            print(f"WARNING! The number of active components for sample {i} is zero. This may indicate an issue with the inner maximization result.")
        
        if num_points > 0:
            # Slice the exact chunk of points belonging to sample i
            new_pts_for_i = final_supports[current_idx : current_idx + num_points]
            
            # Merge into the existing history for sample i
            if len(S_history[i]) == 0:
                S_history[i] = new_pts_for_i
            else:
                S_history[i] = np.vstack([S_history[i], new_pts_for_i])
                
            # Advance the pointer in the flat array
            current_idx += num_points
            
    return S_history


def project_l1_ball(x: np.ndarray, R: float) -> np.ndarray:
    """
    Projects a vector x onto the L1-ball of radius R.
    Uses an exact soft-thresholding algorithm.
    """
    if np.sum(np.abs(x)) <= R:
        return x
    
    # Extract absolute values and sort descending
    u = np.sort(np.abs(x))[::-1]
    cssv = np.cumsum(u)
    
    # Find the threshold index
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - R))[0][-1]
    theta = (cssv[rho] - R) / (rho + 1.0)
    
    # Apply soft-thresholding
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0.0)


def distributional_best_response(z_empirical: np.ndarray, 
                                 problem: SaddlePointProbleml2, 
                                 x_init: np.ndarray, 
                                 lr: Callable[[int], float], 
                                 T: int, 
                                 epsilon: float,
                                 dual_optimal: float = None):
    """
    Distributional Best-Response Algorithm for solving primal Min-Max DRO.
    """
    N, d = z_empirical.shape
    R = problem.R

    # Initialize variables
    x_t = x_init.copy()
    S_history = [z_empirical[i].reshape(1, d) for i in range(N)]  # historical support tracker
    
    # Initialize the high-performance solver
    solver = InnerSolverl2(epsilon=epsilon)

    # --- Tracking Statistics ---
    history = {}
    cumulative_algo_time = 0.0
    burn_in = T // 5  # 20% burn-in period for running averages

    for t in range(1, T + 1):
        if t % 10 == 0:
            print(f"Running iteration {t}/{T}...")

        # === ALGORITHM TIMER START ===
        t_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Inner Maximization (Best Response of the Adversary)
        # Q_t ≈ argmax_Q f(x_t, Q)
        # -------------------------------------------------------------
        loss_t = problem.build_inner_loss(x_t)
        result = solver.solve(z_empirical, loss_t)
        
        # Compress to N+1 points to keep the mixture distribution lightweight
        result_compress = compress_greedy(result, z_empirical, loss_t, epsilon)
        Q_t_weights, Q_t_supports = result_compress.worst_case_distribution
        print(f"Iteration {t}: Inner maximization complete. Worst-case risk = {result_compress.worst_case_loss:.6f}")

        # Update the history of support points for each empirical sample
        S_history = update_support_history(S_history, result_compress, N)
        
        # -------------------------------------------------------------
        # 2. Outer Minimization (Online Algorithm Update)
        # x_{t+1} = A(x_t, f(., Q_t)) -> Gradient Descent
        # -------------------------------------------------------------
        grad_x = problem.expected_gradient_x(x_t, Q_t_weights, Q_t_supports)
        current_lr = lr(t)
        x_next = x_t - current_lr * grad_x

        # Projection step: Ensure x_next remains within the trust region L1-ball
        x_next = project_l1_ball(x_next, R)

        # -------------------------------------------------------------
        # 3. Running Averages (Ergodic Sequence)
        # -------------------------------------------------------------
        if t > burn_in:
            # k is the effective counter for the post-burn-in iterations
            k = t - burn_in 
            
            if k == 1:
                # Initialize the running average 
                x_bar = x_t.copy()
                Q_bar_weights = Q_t_weights.copy()
                Q_bar_supports = Q_t_supports.copy()
            else:
                # Update the running average using the shifted counter
                x_bar = ((k - 1) / k) * x_bar + (1 / k) * x_t

                # Scale old weights by (k-1)/k and new weights by 1/k, then concatenate
                updated_old_weights = ((k - 1) / k) * Q_bar_weights
                scaled_new_weights = (1 / k) * Q_t_weights
                
                Q_bar_weights = np.concatenate([updated_old_weights, scaled_new_weights])
                Q_bar_supports = np.vstack([Q_bar_supports, Q_t_supports])
        
        # Termination check
        prog = np.linalg.norm(x_next - x_t)
        print(f"Progress of x at iteration {t}: {prog:.6f}")

        # Prepare for next iteration
        x_t = x_next

        cumulative_algo_time += (time.perf_counter() - t_start)
        # === ALGORITHM TIMER END ===

        # -------------------------------------------------------------
        # Tracking / Profiling Block (Not counted in algo time)
        # -------------------------------------------------------------
        if t >= T:
            loss_bar = problem.build_inner_loss(x_bar)
            result_bar = solver.solve(z_empirical, loss_bar)
            wcl = result_bar.worst_case_loss
            print(f"Iteration {t}: Evaluated maximum risk for x_bar = {result_bar.worst_case_loss:.6f}")

            x_raw, val_x_raw = problem.best_response_x(Q_bar_weights, Q_bar_supports)

            t_start = time.perf_counter()
            merged_w, merged_s = merge_close_supports(Q_bar_weights, Q_bar_supports, tol=1e-3)
            t_merged = time.perf_counter() - t_start
            x_merged, val_x_merged = problem.best_response_x(merged_w, merged_s)

            t_start = time.perf_counter()
            L_vec, C_vec, G_mat, A_eq, Z_flat, M_total, K = prepare_tangent_compression_data(x_bar, z_empirical, S_history, problem.A, problem.B, problem.C)
            result_tangent = compress_distribution_tangent_quad(x_bar, N, K, epsilon, R, L_vec, C_vec, G_mat, A_eq, Z_flat, M_total)
            t_tangent = time.perf_counter() - t_start
            tangent_w, tangent_s = result_tangent.worst_case_distribution
            x_tangent, val_x_tangent = problem.best_response_x(tangent_w, tangent_s)

            t_start = time.perf_counter()
            C_vec, z_A_z_flat, u_mat_flat, H_list, A_eq, Z_flat, M_total, K = prepare_dual_compression_data(z_empirical, S_history, problem.A, problem.B, problem.C)
            result_dual = compress_distribution_dual_quad(x_bar, N, K, epsilon, R, C_vec, z_A_z_flat, u_mat_flat, H_list, A_eq, Z_flat, M_total)
            t_dual = time.perf_counter() - t_start
            dual_w, dual_s = result_dual.worst_case_distribution
            x_dual, val_x_dual = problem.best_response_x(dual_w, dual_s)

            print(f"Iteration {t}: Raw support size = {len(Q_bar_weights)}, Merged = {len(merged_w)}, Tangent = {len(tangent_w)}, Dual = {len(dual_w)}")
            print(f"Iteration {t}: Best response objective -> Raw: {val_x_raw:.6f}, Merged: {val_x_merged:.6f}, Tangent: {val_x_tangent:.6f}, Dual: {val_x_dual:.6f}")
            print(f"Iteration {t}: Compression time -> Merged: {t_merged:.4f}s, Tangent: {t_tangent:.4f}s, Dual: {t_dual:.4f}s")
            print(f"Accumulated algorithm time = {cumulative_algo_time:.4f} seconds")
            if dual_optimal is not None and dual_optimal != 0:
                print(f"Iteration {t}: Relative suboptimality gap -> Raw: {np.abs((val_x_raw - dual_optimal)/dual_optimal):.6f}, Merged: {np.abs((val_x_merged - dual_optimal)/dual_optimal):.6f}, Tangent: {np.abs((val_x_tangent - dual_optimal)/dual_optimal):.6f}, Dual: {np.abs((val_x_dual - dual_optimal)/dual_optimal):.6f}")
                print(f"Iteration {t}: Relative duality gap -> Raw: {np.abs((val_x_raw - wcl)/dual_optimal):.6f}, Merged: {np.abs((val_x_merged - wcl)/dual_optimal):.6f}, Tangent: {np.abs((val_x_tangent - wcl)/dual_optimal):.6f}, Dual: {np.abs((val_x_dual - wcl)/dual_optimal):.6f}")

            # --- Tracking ---
            history[t] = {
                'risk_x_t': result_compress.worst_case_loss,
                'prog_norm': prog,
                'grad_norm': np.linalg.norm(grad_x),
                'time_cumulative': cumulative_algo_time,
                'risk_x_bar': wcl,
                'supp_size_raw': len(Q_bar_weights),
                'val_loss_raw': val_x_raw,
                'supp_size_merged': len(merged_w),
                'val_loss_merged': val_x_merged,
                'supp_size_tangent': len(tangent_w),
                'val_loss_tangent': val_x_tangent,
                'supp_size_dual': len(dual_w),
                'val_loss_dual': val_x_dual
            }
        else:
            # --- Tracking ---
            history[t] = {
                'risk_x_t': result_compress.worst_case_loss,
                'prog_norm': prog,
                'grad_norm': np.linalg.norm(grad_x),
                'time_cumulative': cumulative_algo_time
            }

        # Termination check based on progress of x_t
        if prog < 1e-5:
            print(f"Distributional Best-Response converged at iteration {t} with progress {prog:.6e}.")
            break
        
    return x_bar, (Q_bar_weights, Q_bar_supports), history


def merge_close_supports(weights: np.ndarray, supports: np.ndarray, tol: float = 1e-5):
    """
    Checks the support of the distribution. If two points are within `tol` 
    distance of each other, they are merged by summing their probability weights.
    """
    if len(weights) == 0:
        return weights, supports
        
    merged_weights = []
    merged_supports = []

    for w, pt in zip(weights, supports):
        found_match = False
        for i, u_pt in enumerate(merged_supports):
            if np.linalg.norm(pt - u_pt, ord=2) < tol:
                # Point is too close! Merge the mass into the existing point
                merged_weights[i] += w
                found_match = True
                break
        
        if not found_match:
            # For any distinct point, add it to the unique list
            merged_weights.append(w)
            merged_supports.append(pt)
            
    return np.array(merged_weights), np.array(merged_supports)