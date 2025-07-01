# -*- coding: utf-8 -*-
"""Python translation of the MATLAB Evolutionary Structural Optimisation code."""
import os
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve
import random
import matplotlib.pyplot as plt
from numba import njit


@njit
def init_image(initial_boundary_conditions, conductive_cells, k0, kp_k0):
    """Fill a blank domain with random conductive cells."""
    # vectorised random fill of the domain instead of a while loop
    candidates = np.argwhere(initial_boundary_conditions == k0)
    if conductive_cells >= len(candidates):
        for idx in range(len(candidates)):
            i = candidates[idx, 0]
            j = candidates[idx, 1]
            initial_boundary_conditions[i, j] = kp_k0
        return initial_boundary_conditions

    perm = np.random.permutation(len(candidates))
    selected = perm[:conductive_cells]
    for idx in selected:
        i = candidates[idx, 0]
        j = candidates[idx, 1]
        initial_boundary_conditions[i, j] = kp_k0
    return initial_boundary_conditions


@njit
def _assemble_system(boundary_conditions, conductivity_table, temp, address, k0, delta_x, p_vol):
    height, width = boundary_conditions.shape
    max_entries = height * width * 5
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=np.float64)
    B = np.zeros(height * width, dtype=np.float64)
    idx = 0
    for i in range(height):
        for j in range(width):
            ind_1 = address[i, j]
            fin = False
            if boundary_conditions[i, j] not in (-2, -3):
                ind_2 = address[i - 1, j]
                ind_3 = address[i, j + 1]
                ind_4 = address[i + 1, j]
                ind_5 = address[i, j - 1]
                somme_cond = conductivity_table[i, j, 0] + conductivity_table[i, j, 1] + conductivity_table[i, j, 2] + conductivity_table[i, j, 3]
            if boundary_conditions[i, j] in (-2, -3):
                B[ind_1] = temp[i, j]
                rows[idx] = ind_1
                cols[idx] = ind_1
                data[idx] = 1.0
                idx += 1
                fin = True
            else:
                rows[idx] = ind_1
                cols[idx] = ind_1
                data[idx] = -somme_cond
                idx += 1
                if boundary_conditions[i, j] == k0:
                    B[ind_1] -= p_vol * delta_x * delta_x
            if not fin:
                if boundary_conditions[i - 1, j] in (-2, -3):
                    B[ind_1] -= temp[i - 1, j] * conductivity_table[i, j, 0]
                else:
                    rows[idx] = ind_1
                    cols[idx] = ind_2
                    data[idx] = conductivity_table[i, j, 0]
                    idx += 1
                if boundary_conditions[i, j + 1] in (-2, -3):
                    B[ind_1] -= temp[i, j + 1] * conductivity_table[i, j, 1]
                else:
                    rows[idx] = ind_1
                    cols[idx] = ind_3
                    data[idx] = conductivity_table[i, j, 1]
                    idx += 1
                if boundary_conditions[i + 1, j] in (-2, -3):
                    B[ind_1] -= temp[i + 1, j] * conductivity_table[i, j, 2]
                else:
                    rows[idx] = ind_1
                    cols[idx] = ind_4
                    data[idx] = conductivity_table[i, j, 2]
                    idx += 1
                if boundary_conditions[i, j - 1] in (-2, -3):
                    B[ind_1] -= temp[i, j - 1] * conductivity_table[i, j, 3]
                else:
                    rows[idx] = ind_1
                    cols[idx] = ind_5
                    data[idx] = conductivity_table[i, j, 3]
                    idx += 1
    return rows[:idx], cols[:idx], data[:idx], B


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
    rows, cols, data, B = _assemble_system(boundary_conditions, conductivity_table, temp, address, k0, delta_x, p_vol)

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
    for (gk, gl) in grow_pos:
        temp_cond = boundary_conditions.copy()
        temp_cond[gk, gl] = kp_k0
        _, _, _, _, _, _, obj, _, _, _ = finite_temp_direct_sparse(
            kp_k0, k0, heat_sink_temperature, delta_x, p_vol, temp_cond
        )
        grow_scores.append((obj, gk, gl))
    grow_scores.sort()

    etch_scores = []
    for (ek, el) in etch_pos:
        temp_cond = boundary_conditions.copy()
        temp_cond[ek, el] = k0
        _, _, _, _, _, _, obj, _, _, _ = finite_temp_direct_sparse(
            kp_k0, k0, heat_sink_temperature, delta_x, p_vol, temp_cond
        )
        etch_scores.append((obj, ek, el))
    etch_scores.sort()

    growth = random.sample(grow_scores[:max_rank], min(max_rank, len(grow_scores)))
    etching = random.sample(etch_scores[:max_rank], min(max_rank, len(etch_scores)))

    for i in range(min(max_cell_swap, len(growth), len(etching))):
        _, gk, gl = growth[i]
        _, ek, el = etching[i]
        boundary_conditions[ek, el] = k0
        boundary_conditions[gk, gl] = kp_k0

    return boundary_conditions, np.array(growth), np.array(etching)


def run_eso_method():
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

        for i in range(min(max_cell_swap, len(growth))):
            history_map[growth[i,1].astype(int), growth[i,2].astype(int)] += 1
            history_map[etching[i,1].astype(int), etching[i,2].astype(int)] += 1

        if history_map.max() > history_map.max():
            max_cell_swap = max(max_cell_swap - 1, 1)

        if m > 10:  # limit iterations for testing
            break

    print('Converged after', m, 'iterations')


if __name__ == '__main__':
    run_eso_method()
