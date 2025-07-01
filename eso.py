# -*- coding: utf-8 -*-
"""Python translation of the MATLAB Evolutionary Structural Optimisation code."""
import os
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve
import random


def init_image(initial_boundary_conditions, conductive_cells, k0, kp_k0):
    """Fill a blank domain with random conductive cells."""
    height, width = initial_boundary_conditions.shape
    k = 0
    while k < conductive_cells:
        h = np.random.randint(0, height)
        l = np.random.randint(0, width)
        if initial_boundary_conditions[h, l] == k0:
            initial_boundary_conditions[h, l] = kp_k0
            k += 1
    return initial_boundary_conditions


def finite_temp_direct_sparse(kp_k0, k0, heat_sink_temperature, delta_x, p_vol, boundary_conditions):
    """Direct finite difference solver translated from MATLAB."""
    height, width = boundary_conditions.shape
    temp = np.ones((height, width)) * heat_sink_temperature
    conductivity_table = np.zeros((height, width, 4))
    temporary = boundary_conditions.copy()
    temporary[temporary == -2] = 1e-9
    temporary[temporary == -3] = kp_k0

    for k in range(1, height-1):
        for l in range(1, width-1):
            conductivity_table[k, l, 0] = delta_x / ((delta_x / 2) / temporary[k-1, l] + (delta_x / 2) / temporary[k, l])
            conductivity_table[k, l, 1] = delta_x / ((delta_x / 2) / temporary[k, l+1] + (delta_x / 2) / temporary[k, l])
            conductivity_table[k, l, 2] = delta_x / ((delta_x / 2) / temporary[k+1, l] + (delta_x / 2) / temporary[k, l])
            conductivity_table[k, l, 3] = delta_x / ((delta_x / 2) / temporary[k, l-1] + (delta_x / 2) / temporary[k, l])

    address = np.arange(height * width).reshape((height, width))
    rows, cols, data = [], [], []
    B = np.zeros(height * width)

    for i in range(height):
        for j in range(width):
            ind_1 = address[i, j]
            if boundary_conditions[i, j] not in (-2, -3):
                ind_2 = address[i - 1, j]
                ind_3 = address[i, j + 1]
                ind_4 = address[i + 1, j]
                ind_5 = address[i, j - 1]
                somme_cond = conductivity_table[i, j].sum()
            fin = False
            if boundary_conditions[i, j] in (-2, -3):
                B[ind_1] = temp[i, j]
                rows.append(ind_1)
                cols.append(ind_1)
                data.append(1)
                fin = True
            else:
                rows.append(ind_1)
                cols.append(ind_1)
                data.append(-somme_cond)
                if boundary_conditions[i, j] == k0:
                    B[ind_1] -= p_vol * delta_x**2
            if not fin:
                if boundary_conditions[i - 1, j] in (-2, -3):
                    B[ind_1] -= temp[i - 1, j] * conductivity_table[i, j, 0]
                else:
                    rows.append(ind_1)
                    cols.append(ind_2)
                    data.append(conductivity_table[i, j, 0])
                if boundary_conditions[i, j + 1] in (-2, -3):
                    B[ind_1] -= temp[i, j + 1] * conductivity_table[i, j, 1]
                else:
                    rows.append(ind_1)
                    cols.append(ind_3)
                    data.append(conductivity_table[i, j, 1])
                if boundary_conditions[i + 1, j] in (-2, -3):
                    B[ind_1] -= temp[i + 1, j] * conductivity_table[i, j, 2]
                else:
                    rows.append(ind_1)
                    cols.append(ind_4)
                    data.append(conductivity_table[i, j, 2])
                if boundary_conditions[i, j - 1] in (-2, -3):
                    B[ind_1] -= temp[i, j - 1] * conductivity_table[i, j, 3]
                else:
                    rows.append(ind_1)
                    cols.append(ind_5)
                    data.append(conductivity_table[i, j, 3])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(height * width, height * width))
    T = spsolve(A, B)

    for i in range(height):
        for j in range(width):
            temp[i, j] = T[address[i, j]]

    maximal_temperature = temp.max()
    variance = temp.var()
    grad_x, grad_y = np.gradient(temp[1:-1, 1:-1])
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    variance_grad = grad.var()
    mean_temperature = temp.mean()
    border_variance = np.std(np.concatenate((temp[1:-1, 1], temp[1, 1:-1])))

    entropy = np.zeros((height, width))
    sum_of_entropy = 0.0
    for k in range(height):
        for l in range(width):
            if boundary_conditions[k, l] in (-2, -3):
                continue
            flux1 = conductivity_table[k, l, 0] * (temp[k - 1, l] - temp[k, l])
            flux2 = conductivity_table[k, l, 1] * (temp[k, l + 1] - temp[k, l])
            flux3 = conductivity_table[k, l, 2] * (temp[k + 1, l] - temp[k, l])
            flux4 = conductivity_table[k, l, 3] * (temp[k, l - 1] - temp[k, l])
            flux5 = p_vol * delta_x * delta_x if boundary_conditions[k, l] == k0 else 0.0
            if boundary_conditions[k - 1, l] != -2:
                entropy[k, l] += (abs(flux1 / temp[k, l] - abs(flux1 / temp[k - 1, l]))) * np.sign(temp[k, l] - temp[k - 1, l])
            if boundary_conditions[k, l + 1] != -2:
                entropy[k, l] += (abs(flux2 / temp[k, l] - abs(flux2 / temp[k, l + 1]))) * np.sign(temp[k, l] - temp[k, l + 1])
            if boundary_conditions[k + 1, l] != -2:
                entropy[k, l] += (abs(flux3 / temp[k, l] - abs(flux3 / temp[k + 1, l]))) * np.sign(temp[k, l] - temp[k + 1, l])
            if boundary_conditions[k, l - 1] != -2:
                entropy[k, l] += (abs(flux4 / temp[k, l] - abs(flux4 / temp[k, l - 1]))) * np.sign(temp[k, l] - temp[k, l - 1])
            entropy[k, l] += flux5 / temp[k, l]
            sum_of_entropy += entropy[k, l]

    row, col = np.argwhere(temp == maximal_temperature)[0]
    row_ref = height - 1
    col_ref = width - 1
    distance = (row - row_ref) ** 2 + (col - col_ref) ** 2

    return distance, sum_of_entropy, entropy, border_variance, variance, mean_temperature, maximal_temperature, temp, grad, variance_grad


def fun_eso_algorithm(boundary_conditions, kp_k0, k0, heat_sink_temperature, delta_x, p_vol, max_rank, max_cell_swap):
    height, width = boundary_conditions.shape
    grow = np.zeros((height, width), dtype=bool)
    for k in range(1, height-1):
        for l in range(1, width-1):
            if boundary_conditions[k, l] == kp_k0:
                if boundary_conditions[k+1, l] == k0:
                    grow[k+1, l] = True
                if boundary_conditions[k-1, l] == k0:
                    grow[k-1, l] = True
                if boundary_conditions[k, l+1] == k0:
                    grow[k, l+1] = True
                if boundary_conditions[k, l-1] == k0:
                    grow[k, l-1] = True

    grow_pos = [(k, l) for k in range(height) for l in range(width) if grow[k, l]]

    etch = np.zeros((height, width), dtype=bool)
    for k in range(1, height-1):
        for l in range(1, width-1):
            if boundary_conditions[k, l] == k0:
                if boundary_conditions[k+1, l] == kp_k0:
                    etch[k+1, l] = True
                if boundary_conditions[k-1, l] == kp_k0:
                    etch[k-1, l] = True
                if boundary_conditions[k, l+1] == kp_k0:
                    etch[k, l+1] = True
                if boundary_conditions[k, l-1] == kp_k0:
                    etch[k, l-1] = True

    etch_pos = [(k, l) for k in range(height) for l in range(width) if etch[k, l]]

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

    boundary_image = Image.open(starting_image)
    initial_bc = np.array(boundary_image)
    height, width, _ = initial_bc.shape
    boundary_conditions = np.zeros((height, width))
    history_map = np.zeros((height, width))
    history_map[0, 0] = 1

    non_conductive = 0
    conductive = 0
    for k in range(height):
        for l in range(width):
            red, green, blue = initial_bc[k, l]
            if red == 255 and green == 255 and blue == 255:
                pixel = low_conductivity
                non_conductive += 1
            elif red == 127 and green == 127 and blue == 127:
                pixel = -2
            elif red == 0 and green == 0 and blue == 255:
                pixel = -3
            else:
                pixel = high_conductivity
                conductive += 1
            boundary_conditions[k, l] = pixel

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
