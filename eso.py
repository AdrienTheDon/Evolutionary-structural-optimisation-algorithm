# -*- coding: utf-8 -*-
"""Python translation of the MATLAB Evolutionary Structural Optimisation code."""
import os
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def init_image(initial_boundary_conditions, conductive_cells, k0, kp_k0):
    """Fill a blank domain with random conductive cells."""
    # vectorised random fill of the domain instead of a while loop
    candidates = np.argwhere(initial_boundary_conditions == k0)
    if conductive_cells >= len(candidates):
        initial_boundary_conditions[candidates[:, 0], candidates[:, 1]] = kp_k0
        return initial_boundary_conditions

    chosen = np.random.choice(len(candidates), conductive_cells, replace=False)
    selected = candidates[chosen]
    initial_boundary_conditions[selected[:, 0], selected[:, 1]] = kp_k0
    return initial_boundary_conditions


def finite_temp_direct_sparse(kp_k0, k0, heat_sink_temperature, delta_x, p_vol, boundary_conditions):
    """Direct finite difference solver translated from MATLAB."""
    height, width = boundary_conditions.shape
    temp = np.ones((height, width)) * heat_sink_temperature
    conductivity_table = np.zeros((height, width, 4))
    temporary = boundary_conditions.copy()
    temporary[temporary == -2] = 1e-9
    temporary[temporary == -3] = kp_k0

    center = temporary[1:-1, 1:-1]
    north = temporary[:-2, 1:-1]
    east = temporary[1:-1, 2:]
    south = temporary[2:, 1:-1]
    west = temporary[1:-1, :-2]

    conductivity_table[1:-1, 1:-1, 0] = delta_x / ((delta_x / 2) / north + (delta_x / 2) / center)
    conductivity_table[1:-1, 1:-1, 1] = delta_x / ((delta_x / 2) / east + (delta_x / 2) / center)
    conductivity_table[1:-1, 1:-1, 2] = delta_x / ((delta_x / 2) / south + (delta_x / 2) / center)
    conductivity_table[1:-1, 1:-1, 3] = delta_x / ((delta_x / 2) / west + (delta_x / 2) / center)

    address = np.arange(height * width).reshape((height, width))
    B = np.zeros(height * width)

    dirichlet_mask = (boundary_conditions == -2) | (boundary_conditions == -3)

    # indices of all cells
    center_idx = address[1:-1, 1:-1]

    # mask of internal cells (non Dirichlet)
    internal_mask = ~dirichlet_mask[1:-1, 1:-1]

    # diagonal terms for Dirichlet cells
    dir_idx = address[dirichlet_mask]
    rows_dir = dir_idx
    cols_dir = dir_idx
    data_dir = np.ones_like(dir_idx, dtype=float)
    B[dir_idx] = temp[dirichlet_mask]

    # diagonal terms for internal cells
    diag_idx = center_idx[internal_mask]
    somme_cond = conductivity_table[1:-1, 1:-1].sum(axis=2)[internal_mask]
    rows_diag = diag_idx
    cols_diag = diag_idx
    data_diag = -somme_cond

    k0_mask = boundary_conditions[1:-1, 1:-1][internal_mask] == k0
    B[diag_idx[k0_mask]] -= p_vol * delta_x ** 2

    # neighbour indices
    north_idx = address[:-2, 1:-1][internal_mask]
    east_idx = address[1:-1, 2:][internal_mask]
    south_idx = address[2:, 1:-1][internal_mask]
    west_idx = address[1:-1, :-2][internal_mask]

    north_bc = boundary_conditions[:-2, 1:-1][internal_mask]
    east_bc = boundary_conditions[1:-1, 2:][internal_mask]
    south_bc = boundary_conditions[2:, 1:-1][internal_mask]
    west_bc = boundary_conditions[1:-1, :-2][internal_mask]

    cond_n = conductivity_table[1:-1, 1:-1, 0][internal_mask]
    cond_e = conductivity_table[1:-1, 1:-1, 1][internal_mask]
    cond_s = conductivity_table[1:-1, 1:-1, 2][internal_mask]
    cond_w = conductivity_table[1:-1, 1:-1, 3][internal_mask]

    # masks for neighbour Dirichlet
    mask_n_dir = (north_bc == -2) | (north_bc == -3)
    mask_e_dir = (east_bc == -2) | (east_bc == -3)
    mask_s_dir = (south_bc == -2) | (south_bc == -3)
    mask_w_dir = (west_bc == -2) | (west_bc == -3)

    B[diag_idx[mask_n_dir]] -= temp[:-2, 1:-1][internal_mask][mask_n_dir] * cond_n[mask_n_dir]
    B[diag_idx[mask_e_dir]] -= temp[1:-1, 2:][internal_mask][mask_e_dir] * cond_e[mask_e_dir]
    B[diag_idx[mask_s_dir]] -= temp[2:, 1:-1][internal_mask][mask_s_dir] * cond_s[mask_s_dir]
    B[diag_idx[mask_w_dir]] -= temp[1:-1, :-2][internal_mask][mask_w_dir] * cond_w[mask_w_dir]

    rows_north = diag_idx[~mask_n_dir]
    cols_north = north_idx[~mask_n_dir]
    data_north = cond_n[~mask_n_dir]

    rows_east = diag_idx[~mask_e_dir]
    cols_east = east_idx[~mask_e_dir]
    data_east = cond_e[~mask_e_dir]

    rows_south = diag_idx[~mask_s_dir]
    cols_south = south_idx[~mask_s_dir]
    data_south = cond_s[~mask_s_dir]

    rows_west = diag_idx[~mask_w_dir]
    cols_west = west_idx[~mask_w_dir]
    data_west = cond_w[~mask_w_dir]

    rows = np.concatenate([rows_dir, rows_diag, rows_north, rows_east, rows_south, rows_west])
    cols = np.concatenate([cols_dir, cols_diag, cols_north, cols_east, cols_south, cols_west])
    data = np.concatenate([data_dir, data_diag, data_north, data_east, data_south, data_west])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(height * width, height * width))
    T = spsolve(A, B)

    temp = T.reshape((height, width))

    maximal_temperature = temp.max()
    variance = temp.var()
    grad_x, grad_y = np.gradient(temp[1:-1, 1:-1])
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    variance_grad = grad.var()
    mean_temperature = temp.mean()
    border_variance = np.std(np.concatenate((temp[1:-1, 1], temp[1, 1:-1])))

    entropy = np.zeros((height, width))
    sum_of_entropy = 0.0
    mask_internal = (boundary_conditions != -2) & (boundary_conditions != -3)
    flux1 = conductivity_table[:, :, 0] * (np.roll(temp, 1, axis=0) - temp)
    flux2 = conductivity_table[:, :, 1] * (np.roll(temp, -1, axis=1) - temp)
    flux3 = conductivity_table[:, :, 2] * (np.roll(temp, -1, axis=0) - temp)
    flux4 = conductivity_table[:, :, 3] * (np.roll(temp, 1, axis=1) - temp)
    flux5 = np.where(boundary_conditions == k0, p_vol * delta_x * delta_x, 0.0)

    valid_n = boundary_conditions != -2
    valid_s = np.roll(boundary_conditions, -1, axis=0) != -2
    valid_e = np.roll(boundary_conditions, -1, axis=1) != -2
    valid_w = np.roll(boundary_conditions, 1, axis=1) != -2

    entropy += np.where(valid_n, np.abs(flux1 / temp - np.abs(flux1 / np.roll(temp, 1, axis=0))) * np.sign(temp - np.roll(temp, 1, axis=0)), 0)
    entropy += np.where(valid_e, np.abs(flux2 / temp - np.abs(flux2 / np.roll(temp, -1, axis=1))) * np.sign(temp - np.roll(temp, -1, axis=1)), 0)
    entropy += np.where(valid_s, np.abs(flux3 / temp - np.abs(flux3 / np.roll(temp, -1, axis=0))) * np.sign(temp - np.roll(temp, -1, axis=0)), 0)
    entropy += np.where(valid_w, np.abs(flux4 / temp - np.abs(flux4 / np.roll(temp, 1, axis=1))) * np.sign(temp - np.roll(temp, 1, axis=1)), 0)
    entropy += flux5 / np.where(temp == 0, 1, temp)

    entropy *= mask_internal
    sum_of_entropy = entropy.sum()

    row, col = np.argwhere(temp == maximal_temperature)[0]
    row_ref = height - 1
    col_ref = width - 1
    distance = (row - row_ref) ** 2 + (col - col_ref) ** 2

    return distance, sum_of_entropy, entropy, border_variance, variance, mean_temperature, maximal_temperature, temp, grad, variance_grad


def _evaluate_candidate(boundary_conditions, kp_k0, k0, heat_sink_temperature, delta_x, p_vol, k, l, value):
    """Helper for parallel candidate evaluation."""
    temp_cond = boundary_conditions.copy()
    temp_cond[k, l] = value
    _, _, _, _, _, _, obj, _, _, _ = finite_temp_direct_sparse(
        kp_k0, k0, heat_sink_temperature, delta_x, p_vol, temp_cond
    )
    return obj, k, l


def fun_eso_algorithm(boundary_conditions, kp_k0, k0, heat_sink_temperature, delta_x, p_vol, max_rank, max_cell_swap):
    height, width = boundary_conditions.shape
    grow = np.zeros((height, width), dtype=bool)
    kp_mask = boundary_conditions == kp_k0
    grow[1:, :] |= kp_mask[:-1, :] & (boundary_conditions[1:, :] == k0)
    grow[:-1, :] |= kp_mask[1:, :] & (boundary_conditions[:-1, :] == k0)
    grow[:, 1:] |= kp_mask[:, :-1] & (boundary_conditions[:, 1:] == k0)
    grow[:, :-1] |= kp_mask[:, 1:] & (boundary_conditions[:, :-1] == k0)

    grow_pos = np.argwhere(grow)

    etch = np.zeros((height, width), dtype=bool)
    k0_mask = boundary_conditions == k0
    etch[1:, :] |= k0_mask[:-1, :] & (boundary_conditions[1:, :] == kp_k0)
    etch[:-1, :] |= k0_mask[1:, :] & (boundary_conditions[:-1, :] == kp_k0)
    etch[:, 1:] |= k0_mask[:, :-1] & (boundary_conditions[:, 1:] == kp_k0)
    etch[:, :-1] |= k0_mask[:, 1:] & (boundary_conditions[:, :-1] == kp_k0)

    etch_pos = np.argwhere(etch)

    grow_scores = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _evaluate_candidate,
                boundary_conditions,
                kp_k0,
                k0,
                heat_sink_temperature,
                delta_x,
                p_vol,
                int(gk),
                int(gl),
                kp_k0,
            )
            for gk, gl in grow_pos
        ]
        for fut in futures:
            grow_scores.append(fut.result())
    grow_scores.sort()

    etch_scores = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _evaluate_candidate,
                boundary_conditions,
                kp_k0,
                k0,
                heat_sink_temperature,
                delta_x,
                p_vol,
                int(ek),
                int(el),
                k0,
            )
            for ek, el in etch_pos
        ]
        for fut in futures:
            etch_scores.append(fut.result())
    etch_scores.sort()

    growth = random.sample(grow_scores[:max_rank], min(max_rank, len(grow_scores)))
    etching = random.sample(etch_scores[:max_rank], min(max_rank, len(etch_scores)))

    for i in range(min(max_cell_swap, len(growth), len(etching))):
        _, gk, gl = growth[i]
        _, ek, el = etching[i]
        boundary_conditions[ek, el] = k0
        boundary_conditions[gk, gl] = kp_k0

    return boundary_conditions, np.array(growth), np.array(etching)


def run_eso_method(max_iterations=None):
    high_conductivity = 10
    low_conductivity = 1
    heat_sink_temperature = 298
    delta_x = 0.001
    p_vol = 1e6
    filling_ratio = 0.3
    starting_image = 'Codes/50x100.bmp'
    max_rank = 5
    max_cell_swap = 1
    max_redounding_move_allowed = 50

    os.makedirs('Figure', exist_ok=True)
    os.makedirs('Topology', exist_ok=True)

    plt.ion()
    fig, ax = plt.subplots()

    boundary_image = Image.open(starting_image)
    initial_bc = np.array(boundary_image)
    height, width, _ = initial_bc.shape
    boundary_conditions = np.zeros((height, width))
    history_map = np.zeros((height, width))
    history_map[0, 0] = 1

    frames_dir = os.path.join(
        "Pictures",
        f"frames_kp{high_conductivity}_phi{filling_ratio}_{width}x{height}",
    )
    os.makedirs(frames_dir, exist_ok=True)

    rgb = initial_bc[:, :, :3]
    mask_white = np.all(rgb == 255, axis=2)
    mask_gray = np.all(rgb == 127, axis=2)
    mask_blue = np.all(rgb == np.array([0, 0, 255]), axis=2)
    mask_other = ~(mask_white | mask_gray | mask_blue)

    boundary_conditions[mask_white] = low_conductivity
    boundary_conditions[mask_gray] = -2
    boundary_conditions[mask_blue] = -3
    boundary_conditions[mask_other] = high_conductivity

    non_conductive = mask_white.sum()
    conductive = mask_other.sum()

    number_cond = int(np.ceil(non_conductive * filling_ratio))
    boundary_conditions = init_image(boundary_conditions, number_cond, low_conductivity, high_conductivity)

    m = 0
    history_tmax = []
    while history_map.max() < max_redounding_move_allowed:
        if max_iterations is not None and m >= max_iterations:
            break
        m += 1
        boundary_conditions, growth, etching = fun_eso_algorithm(
            boundary_conditions, high_conductivity, low_conductivity,
            heat_sink_temperature, delta_x, p_vol, max_rank, max_cell_swap
        )
        result = finite_temp_direct_sparse(
            high_conductivity, low_conductivity, heat_sink_temperature,
            delta_x, p_vol, boundary_conditions
        )
        t_max = result[6]
        history_tmax.append(t_max)

        print(f"Iteration {m}: T_max={t_max:.2f}")

        ax.clear()
        display_data = np.zeros_like(boundary_conditions)
        display_data[boundary_conditions == high_conductivity] = 1
        ax.imshow(display_data, cmap='gray')
        ax.set_title(f"Iteration {m} - T_max {t_max:.2f}")
        plt.pause(0.01)

        fig.savefig(
            os.path.join(
                "Figure",
                f"Figure_kp_ko_{high_conductivity}_phi_{filling_ratio}_{m:06d}.png",
            )
        )
        fig.savefig(
            os.path.join(
                frames_dir,
                f"frame_{m:06d}.png",
            )
        )
        Image.fromarray((display_data * 255).astype(np.uint8)).save(
            os.path.join(
                "Topology",
                f"Topology_kp_ko_{high_conductivity}_phi_{filling_ratio}_{m:06d}.png",
            )
        )

        prev_max = history_map.max()

        for i in range(min(max_cell_swap, len(growth))):
            history_map[growth[i,1].astype(int), growth[i,2].astype(int)] += 1
            history_map[etching[i,1].astype(int), etching[i,2].astype(int)] += 1

        new_max = history_map.max()
        if new_max > prev_max:
            max_cell_swap = max(max_cell_swap - 1, 1)

    print('Converged after', m, 'iterations')


if __name__ == '__main__':
    import argparse
    import cProfile

    parser = argparse.ArgumentParser(description='Run the ESO demo')
    parser.add_argument('--profile', action='store_true',
                        help='run the algorithm under cProfile')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='terminate after this many iterations')
    args = parser.parse_args()

    if args.profile:
        cProfile.run(
            f'run_eso_method(max_iterations={args.max_iterations!r})',
            'eso_profile.prof'
        )
    else:
        run_eso_method(max_iterations=args.max_iterations)
