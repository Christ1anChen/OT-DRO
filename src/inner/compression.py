import numpy as np
from wdro_inner.solver import InnerMaxResult


def compress_greedy(result: InnerMaxResult, z_empirical: np.ndarray, loss, epsilon: float) -> InnerMaxResult:
    """
    Compresses a 2N-point worst-case distribution into an (N+1)-point distribution
    using the O(N log N) Fractional Knapsack greedy algorithm.
    """
    N = len(z_empirical)
    active_comps = result.active_components
    
    if active_comps is None:
        raise ValueError("Compression requires `active_components` to be tracked in the InnerMaxResult.")

    weights_array, support_array = result.worst_case_distribution
    
    # Track the remaining budget for the knapsack problem: N*rho - sum(c_i^-)
    rho_tilde = N * epsilon
    
    items = []
    ptr = 0  # Pointer to track our position in the flattened distribution arrays
    
    # ---------------------------------------------------------
    # 1. O(N) Extraction using active_components mapping
    # ---------------------------------------------------------
    for i in range(N):
        active_k = active_comps[i]
        num_active = len(active_k)
        
        if num_active == 0:
            # Fallback: No perturbation
            c = 0.0
            l = loss.evaluate(z_empirical[i])
            items.append({
                'i': i, 'is_split': False, 'k': None,
                'z': z_empirical[i], 'cost': c, 'loss': l
            })
            
        elif num_active == 1:
            # Single perturbation
            k = active_k[0]
            
            # Instantly read the pre-calculated position!
            z_new = support_array[ptr]
            ptr += 1
            
            c = np.linalg.norm(z_new - z_empirical[i], ord=2)
            l = loss.components[k].evaluate(z_new)
            
            rho_tilde -= c
            items.append({
                'i': i, 'is_split': False, 'k': k,
                'z': z_new, 'cost': c, 'loss': l
            })
            
        elif num_active == 2:
            # Split perturbation
            k1, k2 = active_k[0], active_k[1]
            
            # Instantly read both pre-calculated positions!
            z1 = support_array[ptr]
            z2 = support_array[ptr + 1]
            ptr += 2
            
            c1 = np.linalg.norm(z1 - z_empirical[i], ord=2)
            c2 = np.linalg.norm(z2 - z_empirical[i], ord=2)
            
            l1 = loss.components[k1].evaluate(z1)
            l2 = loss.components[k2].evaluate(z2)
            
            # Assign (+) to the higher cost point per Theorem requirements
            if c1 >= c2:
                c_plus, c_minus = c1, c2
                l_plus, l_minus = l1, l2
                z_plus, z_minus = z1, z2
                k_plus, k_minus = k1, k2
            else:
                c_plus, c_minus = c2, c1
                l_plus, l_minus = l2, l1
                z_plus, z_minus = z2, z1
                k_plus, k_minus = k2, k1
                
            c_tilde = c_plus - c_minus
            l_tilde = l_plus - l_minus
            
            # Deduct the base cost c_i^- from the total global budget
            rho_tilde -= c_minus
            
            # Calculate Knapsack efficiency ratio: value / weight
            if c_tilde > 1e-12:
                ratio = l_tilde / c_tilde 
            else:
                ratio = np.inf if l_tilde > 0 else -np.inf
            
            items.append({
                'i': i, 'is_split': True,
                'z_plus': z_plus, 'c_plus': c_plus, 'l_plus': l_plus, 'k_plus': k_plus,
                'z_minus': z_minus, 'c_minus': c_minus, 'l_minus': l_minus, 'k_minus': k_minus,
                'c_tilde': c_tilde, 'l_tilde': l_tilde,
                'ratio': ratio
            })

    # Ensure numerical precision doesn't push the budget negative
    rho_tilde = max(0.0, rho_tilde)

    # ---------------------------------------------------------
    # 2. Sorting-Based Greedy Algorithm (Fractional Knapsack)
    # ---------------------------------------------------------
    split_items = [item for item in items if item['is_split']]
    split_items.sort(key=lambda x: x['ratio'], reverse=True)
    
    for item in split_items:
        # If the ratio is negative, spending budget decreases the loss. 
        if item['ratio'] <= 0:
            item['alpha_plus'] = 0.0
            continue

        if rho_tilde >= item['c_tilde'] - 1e-10:
            # We can afford the full step to the higher-loss point
            alpha_plus = 1.0
            rho_tilde -= item['c_tilde']
        elif rho_tilde > 1e-10:
            # We can only afford a fraction (This happens AT MOST ONCE globally)
            alpha_plus = rho_tilde / item['c_tilde']
            rho_tilde = 0.0
        else:
            # Out of budget, must default to the lower-cost point
            alpha_plus = 0.0
            
        item['alpha_plus'] = alpha_plus

    # ---------------------------------------------------------
    # 3. Reconstruction of the N+1 Distribution
    # ---------------------------------------------------------
    final_weights = []
    final_supports = []
    final_active_comps = []
    reconstructed_loss = 0.0
    new_budgets = np.zeros(N)
    
    for item in items:
        current_active = []
        i = item['i']

        if not item['is_split']:
            final_weights.append(1.0 / N)
            final_supports.append(item['z'])
            reconstructed_loss += item['loss'] / N
            new_budgets[i] = item['cost']

            if item['k'] is not None:
                current_active.append(item['k'])
        else:
            ap = item['alpha_plus']
            am = 1.0 - ap
            new_budgets[i] = ap * item['c_plus'] + am * item['c_minus']
            
            if ap > 1e-8:
                final_weights.append(ap / N)
                final_supports.append(item['z_plus'])
                reconstructed_loss += (ap * item['l_plus']) / N
                current_active.append(item['k_plus'])
                
            if am > 1e-8:
                final_weights.append(am / N)
                final_supports.append(item['z_minus'])
                reconstructed_loss += (am * item['l_minus']) / N
                current_active.append(item['k_minus'])
            
        final_active_comps.append(current_active)
                
    return InnerMaxResult(
        worst_case_loss=reconstructed_loss,
        optimal_lambda=result.optimal_lambda, 
        optimal_budgets=new_budgets,
        worst_case_distribution=(np.array(final_weights), np.array(final_supports)),
        active_components=final_active_comps
    )