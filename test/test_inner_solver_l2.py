import numpy as np
import time
from wdro_inner.solver import InnerMaxResult
from wdro_inner.solver_l2 import InnerSolverl2, numba_phi
from wdro_inner.solver_l2_gurobi import GurobiQuadraticSolver
from wdro_inner.solver_l2_mosek import MosekQuadraticSolver
from wdro_inner.losses import PointwiseMaxLoss, RadialLossComponent
from wdro_inner.compression import compress_greedy


def evaluate_total_cost(result: InnerMaxResult, z_empirical: np.ndarray) -> float:
    """
    Evaluates the real total transportation cost (utilized budget) 
    for a given InnerMaxResult mapped against the empirical dataset.
    
    Args:
        result: The InnerMaxResult object containing the worst-case 
                distribution and active components.
        z_empirical: 2D array of the original empirical data points.
                           
    Returns:
        The exact expected transportation cost (float).
    """
    N = len(z_empirical)
    weights, supports = result.worst_case_distribution
    active_components = result.active_components
    if len(active_components) == 0:
        return 0.0
    
    total_cost = 0.0
    current_idx = 0
    
    for i in range(N):
        num_pts = len(active_components[i])
        
        # Safeguard: Skip if this sample has no active perturbations
        if num_pts == 0:
            continue
            
        # Extract the subset of weights and supports belonging to sample i
        w_i = weights[current_idx : current_idx + num_pts]
        s_i = supports[current_idx : current_idx + num_pts]
        
        # Calculate the L2 distance between perturbed supports and the original empirical point
        dist_i = np.linalg.norm(s_i - z_empirical[i], axis=1)
        
        # Accumulate the expected cost (Probability * Distance)
        total_cost += np.sum(w_i * dist_i)
        
        # Advance the pointer
        current_idx += num_pts
        
    return total_cost


def main():
    # ---------------------------------------------------------
    # 1. Generate Synthetic Empirical Samples
    # ---------------------------------------------------------
    np.random.seed(2026)
    n_samples = 100
    n_dimensions = 500
    K = 3
    
    # Global mean vector mu_hat ~ N(0, I_m)
    mu_hat = np.random.randn(n_dimensions)
    
    # Empirical samples z_i ~ N(mu_hat, I_m)
    z_empirical = np.random.randn(n_samples, n_dimensions) + mu_hat
    
    # ---------------------------------------------------------
    # 2. Define the K-piece Quadratic Loss Components
    # ---------------------------------------------------------
    # l_k(z) = c_k - || A_k z - mu_k||_2^2

    loss_components = []

    # Draw nominal decision variable x_nom ~ N(0, I_m)
    x_nom = np.random.randn(n_dimensions)

    for k in range(K):
        # 2a. Primal DRO matrix generation
        # A_k = (1/m) X_A^T X_A + 0.01 I
        X_A = np.random.randn(n_dimensions, n_dimensions)
        A_k = (X_A.T @ X_A) / n_dimensions + 0.01 * np.eye(n_dimensions)
        
        # C_k = (1/m) X_C^T X_C + 0.01 I
        X_C = np.random.randn(n_dimensions, n_dimensions)
        C_k = (X_C.T @ X_C) / n_dimensions + 0.01 * np.eye(n_dimensions)
        
        # B_k = X_B + M_bias
        X_B = np.random.randn(n_dimensions, n_dimensions)
        M_bias = np.random.randn(n_dimensions, n_dimensions)
        B_k = X_B + M_bias
        
        # 2b. Map to inner problem parameters
        c_k = x_nom.T @ C_k @ x_nom
        b_k = B_k @ x_nom
        
        # 2c. Algebraic conversion for RadialLossComponent
        # Target form: l(z) = c_k + b_k^T z - z^T A_k z
        # Component form: intercept - || A_rad z + b_rad ||^2
        #               = intercept - z^T(A_rad^T A_rad)z - 2 b_rad^T A_rad z - ||b_rad||^2
        
        # Match A_rad^T A_rad = A_k via symmetric square root (A_k^{1/2})
        eigenvalues, eigenvectors = np.linalg.eigh(A_k)
        A_rad = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T  
        
        # Match -2 A_rad^T b_rad = b_k  =>  -2 A_rad b_rad = b_k
        b_rad = -0.5 * np.linalg.solve(A_rad, b_k)
        
        # Match intercept - ||b_rad||^2 = c_k
        intercept = c_k + np.sum(b_rad**2)
        
        # Instantiate the component exactly as required by the solver architecture
        comp_k = RadialLossComponent(A=A_rad, b_vec=b_rad, phi=numba_phi, intercept=intercept)
        loss_components.append(comp_k)
    
    # Combine into the Pointwise Max Loss
    loss = PointwiseMaxLoss(loss_components)
    
    # ---------------------------------------------------------
    # 3. Initialize the Solvers
    # ---------------------------------------------------------
    epsilon = 0.1  # The Wasserstein radius
    
    # Lock the metric to L2
    solver = InnerSolverl2(epsilon=epsilon)
    gurobi_solver = GurobiQuadraticSolver(epsilon=epsilon, verbose=False)
    mosek_solver = MosekQuadraticSolver(epsilon=epsilon, verbose=False)

    # ---------------------------------------------------------
    # 4. BENCHMARK: Gurobi
    # ---------------------------------------------------------
    print("Running Gurobi Exact Solver...")
    try:
        t0_gurobi = time.perf_counter()
        result_gurobi = gurobi_solver.solve(z_empirical, loss)
        t1_gurobi = time.perf_counter()
        time_gurobi = t1_gurobi - t0_gurobi
    except Exception as e:
        print(f"Gurobi solver failed with error: {e}")
        result_gurobi = InnerMaxResult(worst_case_loss=np.nan, 
                                       optimal_lambda=np.nan, 
                                       optimal_budgets=np.array([]), 
                                       worst_case_distribution=(np.array([]), np.array([])), 
                                       active_components=np.array([]))
        time_gurobi = np.nan

    # ---------------------------------------------------------
    # 5. BENCHMARK: MOSEK
    # ---------------------------------------------------------
    print("Running MOSEK Exact Solver...")
    try:
        t0_mosek = time.perf_counter()
        result_mosek = mosek_solver.solve(z_empirical, loss)
        t1_mosek = time.perf_counter()
        time_mosek = t1_mosek - t0_mosek
    except Exception as e:
        print(f"MOSEK solver failed with error: {e}")
        result_mosek = InnerMaxResult(worst_case_loss=np.nan, 
                                      optimal_lambda=np.nan, 
                                      optimal_budgets=np.array([]), 
                                      worst_case_distribution=(np.array([]), np.array([])), 
                                      active_components=np.array([]))
        time_mosek = np.nan

    # ---------------------------------------------------------
    # 6. BENCHMARK: Our Custom Algorithm
    # ---------------------------------------------------------
    print("Running Custom L2 Solver...")
    t0_ours = time.perf_counter()
    result_ours = solver.solve(z_empirical, loss)
    t1_ours = time.perf_counter()
    time_ours = t1_ours - t0_ours

    t0_compress = time.perf_counter()
    result_compressed = compress_greedy(result_ours, z_empirical, loss, epsilon)
    t1_compress = time.perf_counter()
    time_compress = t1_compress - t0_compress

    # ---------------------------------------------------------
    # 7. Print Comparison
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    print(f"Custom Algorithm Loss:   {result_ours.worst_case_loss:.8f}")
    print(f"Gurobi Exact Loss:       {result_gurobi.worst_case_loss:.8f}")
    print(f"MOSEK Exact Loss:        {result_mosek.worst_case_loss:.8f}")
    print(f"Compress Algorithm Loss: {result_compressed.worst_case_loss:.8f}")
    
    print("-" * 50)

    print(f"Custom Algorithm Optimal Dual Variable (lambda): {result_ours.optimal_lambda:.8f}")
    print(f"Gurobi Exact Optimal Dual Variable (lambda):     {result_gurobi.optimal_lambda:.8f}")
    print(f"MOSEK Exact Optimal Dual Variable (lambda):      {result_mosek.optimal_lambda:.8f}")

    print("-" * 50)

    print(f"Custom Algorithm Total Cost:   {evaluate_total_cost(result_ours, z_empirical):.8f}")
    print(f"Gurobi Exact Total Cost:       {evaluate_total_cost(result_gurobi, z_empirical):.8f}")
    print(f"MOSEK Exact Total Cost:        {evaluate_total_cost(result_mosek, z_empirical):.8f}")
    print(f"Compress Algorithm Total Cost: {evaluate_total_cost(result_compressed, z_empirical):.8f}")

    print("-" * 50)

    # Extract weights to count atoms (support points)
    weights_ours, _ = result_ours.worst_case_distribution
    print(f"Total Weight (Custom): {weights_ours.sum():.8f}")
    weights_gurobi, _ = result_gurobi.worst_case_distribution
    weights_mosek, _ = result_mosek.worst_case_distribution
    print(f"Total Weight (MOSEK): {weights_mosek.sum():.8f}")
    weights_compressed, _ = result_compressed.worst_case_distribution
    print(f"Total Weight (Compressed): {weights_compressed.sum():.8f}")
    atoms_ours = len(weights_ours)
    atoms_gurobi = len(weights_gurobi)
    atoms_mosek = len(weights_mosek)
    atoms_compressed = len(weights_compressed)

    print(f"Custom Algorithm Atoms:   {atoms_ours}")
    print(f"Gurobi Exact Atoms:       {atoms_gurobi}")
    print(f"MOSEK Exact Atoms:        {atoms_mosek}")
    print(f"Compress Algorithm Atoms: {atoms_compressed}")

    print("-" * 50)

    print(f"Custom Algorithm Active Components per Sample:   {result_ours.active_components}")
    print(f"Gurobi Exact Active Components per Sample:       {result_gurobi.active_components}")
    print(f"MOSEK Exact Active Components per Sample:        {result_mosek.active_components}")
    print(f"Compress Algorithm Active Components per Sample: {result_compressed.active_components}")

    print("-" * 50)

    print(f"Custom Algorithm Optimal Budgets:   [{', '.join(f'{b:.6f}' for b in result_ours.optimal_budgets)}]")
    print(f"Gurobi Exact Optimal Budgets:       [{', '.join(f'{b:.6f}' for b in result_gurobi.optimal_budgets)}]")
    print(f"MOSEK Exact Optimal Budgets:        [{', '.join(f'{b:.6f}' for b in result_mosek.optimal_budgets)}]")
    print(f"Compress Algorithm Optimal Budgets: [{', '.join(f'{b:.6f}' for b in result_compressed.optimal_budgets)}]")

    print("-" * 50)

    print(f"Custom Algorithm Time:   {time_ours:.4f} seconds")
    print(f"Gurobi Exact Time:       {time_gurobi:.4f} seconds")
    print(f"MOSEK Exact Time:        {time_mosek:.4f} seconds")
    print(f"Compress Algorithm Time: {time_compress:.4f} seconds")

if __name__ == "__main__":
    main()