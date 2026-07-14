
from jinja2 import Template
import numpy as np 
import os
from pathlib import Path
import time
import subprocess
import psutil
import shutil

def wait_for_cpu_usage_below(threshold, check_interval=10):
    """
    Pause execution until the CPU usage is below the specified threshold.
    
    :param threshold: The CPU usage percentage to wait below.
    :param check_interval: How many seconds to wait before checking again.
    """
    while True:
        usage = psutil.cpu_percent(interval=30)
        if usage < threshold:
            break
        print(f"CPU usage is {usage}%, waiting {check_interval} seconds to reduce load.")
        time.sleep(check_interval)


def run_in_background(software_path, input_file, output_dir):
    """
    Run a software executable in the background using nohup, and
    save its output to a file in output_dir.

    :param software_path: Full path to the software executable.
    :param input_file:    Path to the input file or script the software needs to process.
    :param output_dir:    Directory where the output (stdout & stderr) will be saved.
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct an output file name. You can modify this as needed.
    output_file = os.path.join(output_dir, "nohup_output.log")
    original_dir = os.getcwd()

    try:
        os.chdir(output_dir)
        
        # output/error redirected to nohup_output.log
        # & to run in background
        command = f"nohup {software_path} {input_file} > {output_file} 2>&1 & echo $!"
        
        #subprocess.run(command, shell=True, check=True)
        output = subprocess.check_output(command, shell=True, cwd=output_dir)
        
        # Convert bytes -> string and remove any trailing whitespace
        pid = output.decode().strip()
        print(f"Started background process in '{output_dir}' with PID: {pid}")
        print(f"Started process in background. Output is going to: {output_file}")
        return pid

    finally:
        # Change back to the original working directory
        os.chdir(original_dir)



ellipse_template = """
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import time
import numpy as np
from numba import njit, prange, set_num_threads

try:
    import tqdm
except Exception: 
    tqdm = None
    
try:
    import h5py
except Exception:
    h5py= None



@dataclass
class SimConfig:
    # ---------------- physical / simulation parameters ----------------
    phi: float = {{phi}}
    dt: float = {{dt}}
    tot_time: float = {{sim_time}}
    L: float = {{boxl}}
    v0: float = {{act_vel}}
    D: float = {{Diff_t}}
    D_r: float = {{Diff_r}}

    # Morse parameters
    a_mor: float = {{a_morse}}
    D_mor: float = {{D_morse}}

    # Harmonic relaxation parameters
    epsilon: float = {{gb_epsilon}}
    beta: float = {{beta_exp}}

    # Deformable-particle parameters
    R_0: float = {{R_0}}
    sigma_0: float = {{sigma_0}}
    aspect_ratio: float = {{aspect_ratio}}
    aspect_ratio_init: float = {{aspect_ratio_init}}
    tau: float = {{KV_tau}}
    mu: float = {{KV_mu}}
    K: float = {{KV_K}}  
    incom_K: bool = {{incom_K}}

    deformable: bool = {{deform_flag}}
    deformable_relax: bool = {{deform_relax_flag}}
    dt_relax: float = {{dt_relax}}
    relax_steps: int = {{relax_steps}}
    shape_diff_tol = {{shape_diff_tol}}

    # other parameters
    seed: int = {{seed}}
    interval: int = {{save_interval}}
    num_threads: int = {{num_threads}}

    # Cell-list neighbour cutoff
    neighbour_cutoff_factor: float = {{neighbour_list_cutoff}}

    # Output controls
    save_data: bool = {{save_data}}
    save_hdf5: bool = {{save_h5}}
    make_movie: bool = {{make_movie}}
    output_prefix: str = "data_ellipses"
    movie_fps: int = {{movie_fps}}
    movie_dpi: int = {{movie_dpi}}


# Edit this object to change defaults from the command line script.
CONFIG = SimConfig()

# Small scalar helpers used by CPU Numba kernels

@njit(inline="always")
def pbc_delta(dx: float, L: float) -> float:
    # Minimum-image displacement for a square periodic box.
    return dx - L * np.rint(dx / L)

@njit(inline="always")
def wrap_position(x: float, L: float) -> float:
    #Wrap coordinate to approximately [-L/2, L/2]
    return x - L * np.rint(x / L)

@njit(inline="always")
def clamp_cell_index(c: int, n_cells: int) -> int:
    if c < 0:
        return 0
    if c >= n_cells:
        return n_cells - 1
    return c

@njit(inline="always")
def particle_cell_indices(xi: float, yi: float, L: float, cell_size: float, n_cells: int):
    cx = int((xi + 0.5 * L) / cell_size)
    cy = int((yi + 0.5 * L) / cell_size)
    return clamp_cell_index(cx, n_cells), clamp_cell_index(cy, n_cells)


@njit
def build_cell_list_cpu(x, y, L: float, n_cells: int, cell_head, cell_next):
    # Build linked-cell list in O(N).

    n = x.shape[0]
    n_total_cells = n_cells * n_cells
    cell_size = L / n_cells

    for c in range(n_total_cells):
        cell_head[c] = -1
    for i in range(n):
        cell_next[i] = -1

    for i in range(n):
        cx, cy = particle_cell_indices(x[i], y[i], L, cell_size, n_cells)
        cid = cx + n_cells * cy
        cell_next[i] = cell_head[cid]
        cell_head[cid] = i


@njit(inline="always")
def calculate_force_torque_morse_scalar(rx: float, ry: float, r: float, chi: float, u1x: float, u1y: float, u2x: float, u2y: float, epsilon: float, a_mor: float, D_mor: float, sigma_0: float,):
    
    rdotu1 = rx * u1x + ry * u1y
    rdotu2 = rx * u2x + ry * u2y
    u1dotu2 = u1x * u2x + u1y * u2y

    dots_add = rdotu1 + rdotu2
    dots_sub = rdotu1 - rdotu2
    chi_add = 1.0 + chi * u1dotu2
    chi_sub = 1.0 - chi * u1dotu2

    r2 = r * r
    sigma_arg = 1.0 - 0.5 * chi / r2 * (dots_add * dots_add / chi_add + dots_sub * dots_sub / chi_sub)
    sigma = sigma_0 * sigma_arg ** (-0.5)

    Fx = 0.0
    Fy = 0.0
    torque = 0.0

    if r < 2.0 * sigma:
        exp_term = math.exp(-a_mor * (r - sigma))
        common_force = 2.0 * a_mor * D_mor * epsilon * exp_term * (1.0 - exp_term)
        common_shape = dots_add * dots_add / chi_add + dots_sub * dots_sub / chi_sub

        geom_x = (rx / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2) * (-rx / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1x + u2x) + dots_sub / chi_sub * (u1x - u2x)) ))
        geom_y = (ry / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2) * (-ry / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1y + u2y) + dots_sub / chi_sub * (u1y - u2y)) ))

        Fx = common_force * geom_x
        Fy = common_force * geom_y

        common_du = ( 0.5 * a_mor * D_mor * epsilon * chi * sigma_0 * exp_term * (1.0 - exp_term) * sigma_arg ** (-1.5) )

        du_geom_x = ( 2.0 * dots_add * rx / r / chi_add + dots_add * dots_add * chi * u2x / (chi_add ** 2) + 2.0 * dots_sub * rx / r / chi_sub + dots_sub * dots_sub * chi * u2x / (chi_sub ** 2) )
        du_geom_y = ( 2.0 * dots_add * ry / r / chi_add + dots_add * dots_add * chi * u2y / (chi_add ** 2) + 2.0 * dots_sub * ry / r / chi_sub + dots_sub * dots_sub * chi * u2y / (chi_sub ** 2) )

        dUdux = common_du * du_geom_x
        dUduy = common_du * du_geom_y
        torque = u1x * dUduy - u1y * dUdux

    return Fx, Fy, torque


@njit(inline="always")
def calculate_force_torque_harmonic_scalar(rx: float, ry: float, r: float, chi: float, u1x: float, u1y: float, u2x: float, u2y: float, epsilon: float, beta: float, sigma_0: float,):
    
    rdotu1 = rx * u1x + ry * u1y
    rdotu2 = rx * u2x + ry * u2y
    u1dotu2 = u1x * u2x + u1y * u2y

    dots_add = rdotu1 + rdotu2
    dots_sub = rdotu1 - rdotu2
    chi_add = 1.0 + chi * u1dotu2
    chi_sub = 1.0 - chi * u1dotu2

    r2 = r * r
    sigma_arg = 1.0 - 0.5 * chi / r2 * (dots_add * dots_add / chi_add + dots_sub * dots_sub / chi_sub)
    sigma = sigma_0 * sigma_arg ** (-0.5)

    Fx = 0.0
    Fy = 0.0
    torque = 0.0

    if r < sigma:
        common_force = 0.5 * epsilon * beta * (r - sigma) ** (beta - 1.0)
        common_shape = dots_add * dots_add / chi_add + dots_sub * dots_sub / chi_sub

        geom_x = (rx / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2) * (-rx / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1x + u2x) + dots_sub / chi_sub * (u1x - u2x)) ))
        geom_y = (ry / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2) * (-ry / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1y + u2y) + dots_sub / chi_sub * (u1y - u2y)) ))

        Fx = common_force * geom_x
        Fy = common_force * geom_y

        common_du = (0.125 * chi * sigma_0 * epsilon * beta * (r - sigma) ** (beta - 1.0) * sigma_arg ** (-1.5))
        du_geom_x = (2.0 * dots_add * rx / r / chi_add + dots_add * dots_add * chi * u2x / (chi_add ** 2) + 2.0 * dots_sub * rx / r / chi_sub + dots_sub * dots_sub * chi * u2x / (chi_sub ** 2) )
        du_geom_y = (2.0 * dots_add * ry / r / chi_add + dots_add * dots_add * chi * u2y / (chi_add ** 2) + 2.0 * dots_sub * ry / r / chi_sub + dots_sub * dots_sub * chi * u2y / (chi_sub ** 2) )

        dUdux = common_du * du_geom_x
        dUduy = common_du * du_geom_y
        torque = u1x * dUduy - u1y * dUdux

    return Fx, Fy, torque



@njit(parallel=True)
def compute_forces_morse_cell_cpu(x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells: int, L: float, cutoff: float, epsilon: float, a_mor: float, D_mor: float,
    sigma_0: float, R_0: float, Fx, Fy, torque_mech, sxx, sxy, syx, syy, ):
    n = x.shape[0]
    cell_size = L / n_cells
    cutoff2 = cutoff * cutoff
    stress_prefactor = 1.0 / (2.0 * math.pi * R_0 * R_0)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        u1x = cos_theta[i]
        u1y = sin_theta[i]
        cxi, cyi = particle_cell_indices(xi, yi, L, cell_size, n_cells)

        fxi = 0.0
        fyi = 0.0
        ti = 0.0
        sxx_i = 0.0
        sxy_i = 0.0
        syx_i = 0.0
        syy_i = 0.0

        for dcx in range(-1, 2):
            cx = (cxi + dcx) % n_cells
            for dcy in range(-1, 2):
                cy = (cyi + dcy) % n_cells
                cid = cx + n_cells * cy
                j = cell_head[cid]
                while j != -1:
                    if j != i:
                        dx = pbc_delta(x[j] - xi, L)
                        dy = pbc_delta(y[j] - yi, L)
                        r2 = dx * dx + dy * dy
                        if 1.0e-24 < r2 < cutoff2:
                            r = math.sqrt(r2)
                            sigma_e_corrected = lmda_major[i] + lmda_major[j]
                            sigma_s_corrected = lmda_minor[i] + lmda_minor[j]
                            ar2 = (sigma_e_corrected / sigma_s_corrected) ** 2
                            chi_corrected = (ar2 - 1.0) / (ar2 + 1.0)

                            fxj, fyj, torque_j = calculate_force_torque_morse_scalar(dx, dy, r, chi_corrected, u1x, u1y, cos_theta[j], sin_theta[j], epsilon, a_mor, D_mor, sigma_0, )
                            fxi += fxj
                            fyi += fyj
                            ti += torque_j
                            sxx_i += dx * fxj
                            sxy_i += dx * fyj
                            syx_i += dy * fxj
                            syy_i += dy * fyj
                    j = cell_next[j]

        Fx[i] = fxi
        Fy[i] = fyi
        torque_mech[i] = ti
        sxx[i] = stress_prefactor * sxx_i
        sxy[i] = stress_prefactor * sxy_i
        syx[i] = stress_prefactor * syx_i
        syy[i] = stress_prefactor * syy_i


@njit(parallel=True)
def compute_forces_harmonic_cell_cpu(
    x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells: int, L: float, cutoff: float, epsilon: float, beta: float, sigma_0: float, R_0: float, Fx, Fy, torque_mech, sxx, sxy, syx, syy, ):
    n = x.shape[0]
    cell_size = L / n_cells
    cutoff2 = cutoff * cutoff
    stress_prefactor = 1.0 / (2.0 * math.pi * R_0 * R_0)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        u1x = cos_theta[i]
        u1y = sin_theta[i]
        cxi, cyi = particle_cell_indices(xi, yi, L, cell_size, n_cells)

        fxi = 0.0
        fyi = 0.0
        ti = 0.0
        sxx_i = 0.0
        sxy_i = 0.0
        syx_i = 0.0
        syy_i = 0.0

        for dcx in range(-1, 2):
            cx = (cxi + dcx) % n_cells
            for dcy in range(-1, 2):
                cy = (cyi + dcy) % n_cells
                cid = cx + n_cells * cy
                j = cell_head[cid]
                while j != -1:
                    if j != i:
                        dx = pbc_delta(x[j] - xi, L)
                        dy = pbc_delta(y[j] - yi, L)
                        r2 = dx * dx + dy * dy
                        if 1.0e-24 < r2 < cutoff2:
                            r = math.sqrt(r2)
                            sigma_e_corrected = lmda_major[i] + lmda_major[j]
                            sigma_s_corrected = lmda_minor[i] + lmda_minor[j]
                            ar2 = (sigma_e_corrected / sigma_s_corrected) ** 2
                            chi_corrected = (ar2 - 1.0) / (ar2 + 1.0)

                            fxj, fyj, torque_j = calculate_force_torque_harmonic_scalar(dx, dy, r, chi_corrected, u1x, u1y, cos_theta[j], sin_theta[j], epsilon, beta, sigma_0, )
                            fxi += fxj
                            fyi += fyj
                            ti += torque_j
                            sxx_i += dx * fxj
                            sxy_i += dx * fyj
                            syx_i += dy * fxj
                            syy_i += dy * fyj
                    j = cell_next[j]

        Fx[i] = fxi
        Fy[i] = fyi
        torque_mech[i] = ti
        sxx[i] = stress_prefactor * sxx_i
        sxy[i] = stress_prefactor * sxy_i
        syx[i] = stress_prefactor * syx_i
        syy[i] = stress_prefactor * syy_i


# State update 

@njit(parallel=True)
def integrate_state_cpu(x, y, theta, Lmda, lmda_major, lmda_minor, Fx, Fy, torque_mech, sxx, sxy, syx, syy, noise_x, noise_y, noise_theta, lmda_major_0: float, lmda_minor_0: float, deformable: bool,
    dt_step: float, L: float, v0: float, D: float, D_r: float, R_0: float, tau: float, mu: float, K:float, incom_K: bool, shape_diff_tol:float, ):
    
    n = x.shape[0]
    tol = shape_diff_tol #1.0e-5
    sqrt_2Ddt = math.sqrt(2.0 * D * dt_step)
    sqrt_2Drdt = math.sqrt(2.0 * D_r * dt_step)

    for i in prange(n):
        th = theta[i]
        si = math.sin(th)
        co = math.cos(th)

        lmaj = lmda_major[i]
        lmin = lmda_minor[i]

        if deformable:
            L00 = Lmda[0, 0, i]
            L01 = Lmda[0, 1, i]
            L10 = Lmda[1, 0, i]
            L11 = Lmda[1, 1, i]

            # Lambda_0 for current orientation and rest lengths.
            L00_0 = co * co * lmda_major_0 + si * si * lmda_minor_0
            L01_0 = si * co * (lmda_major_0 - lmda_minor_0)
            L10_0 = L01_0
            L11_0 = si * si * lmda_major_0 + co * co * lmda_minor_0

            tr = L00 + L11
            det = L00 * L11 - L01 * L10
            delta_lmda = math.sqrt(tr * tr - 4.0 * det) / tr
            mu_dynamic = mu * (1.0 + 50.0 * delta_lmda * delta_lmda)

            if(incom_K):

                dL00 = -(L00 - L00_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (sxx[i] - syy[i])
                dL11 = -(L11 - L11_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (syy[i] - sxx[i])
                dL01 = -(L01 - L01_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * sxy[i]
                dL10 = -(L10 - L10_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * syx[i]
            else:
                dL00 = -(L00 - L00_0) / tau + R_0 / (4.0 * tau * mu_dynamic * K) * ((mu_dynamic + K) * sxx[i] - (mu_dynamic - K) * syy[i])
                dL11 = -(L11 - L11_0) / tau + R_0 / (4.0 * tau * mu_dynamic * K) * ((mu_dynamic + K) * syy[i] - (mu_dynamic - K) * sxx[i])
                dL01 = -(L01 - L01_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * sxy[i]
                dL10 = -(L10 - L10_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * syx[i]

            #dL00 = -(L00 - L00_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (sxx[i] - syy[i])
            #dL11 = -(L11 - L11_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (syy[i] - sxx[i])
            #dL01 = -(L01 - L01_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * sxy[i]
            #dL10 = -(L10 - L10_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * syx[i]

            L00 += dL00 * dt_step
            L11 += dL11 * dt_step
            L01 += dL01 * dt_step
            L10 += dL10 * dt_step

            # Keep Lambda symmetric in the same spirit as the original construction.
            # The original differential update allowed sxy and syx separately, but the
            # following eigenvalue/orientation extraction only uses L01. We preserve
            # both components and reconstruct below from lmaj/lmin/theta.
            Lmda[0, 0, i] = L00
            Lmda[1, 1, i] = L11
            Lmda[0, 1, i] = L01
            Lmda[1, 0, i] = L10

            Ldiff = L00 - L11
            if abs(Ldiff) < tol:
                if Ldiff >= 0.0:
                    Ldiff = tol
                else:
                    Ldiff = -tol

            root = math.sqrt(0.25 * Ldiff * Ldiff + L01 * L01)
            lmaj = 0.5 * (L00 + L11) + root
            lmin = 0.5 * (L00 + L11) - root

            theta_def = 0.5 * math.atan2(2.0 * L01, Ldiff)
            theta_def = theta_def - math.pi * np.rint((theta_def - th) / math.pi)

            th += torque_mech[i] * dt_step + (theta_def - th) + sqrt_2Drdt * noise_theta[i]
            th = wrap_position(th, 2.0 * math.pi)
        else:
            th += torque_mech[i] * dt_step + sqrt_2Drdt * noise_theta[i]
            th = wrap_position(th, 2.0 * math.pi)

        si = math.sin(th)
        co = math.cos(th)

        # Reconstruct Lambda from updated eigenvalues and orientation
        Lmda[0, 0, i] = co * co * lmaj + si * si * lmin
        Lmda[1, 1, i] = si * si * lmaj + co * co * lmin
        Lmda[0, 1, i] = si * co * (lmaj - lmin)
        Lmda[1, 0, i] = Lmda[0, 1, i]

        lmda_major[i] = lmaj
        lmda_minor[i] = lmin
        theta[i] = th

        x[i] = wrap_position(x[i] + (v0 * co + Fx[i]) * dt_step + sqrt_2Ddt * noise_x[i], L)
        y[i] = wrap_position(y[i] + (v0 * si + Fy[i]) * dt_step + sqrt_2Ddt * noise_y[i], L)


# helper functions

def _progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm.tqdm(iterable, desc=desc)


def derived_lengths(cfg: SimConfig):
    sigma_edge = cfg.R_0 * math.sqrt(cfg.aspect_ratio)
    sigma_side = cfg.R_0 / math.sqrt(cfg.aspect_ratio)
    lmda_major_0 = sigma_edge
    lmda_minor_0 = sigma_side
    lmda_major_init = cfg.R_0 * math.sqrt(cfg.aspect_ratio_init)
    lmda_minor_init = cfg.R_0 / math.sqrt(cfg.aspect_ratio_init)
    #lmda_major_init = lmda_major_0 #* cfg.aspect_ratio_init
    #lmda_minor_init = lmda_minor_0
    return lmda_major_0, lmda_minor_0, lmda_major_init, lmda_minor_init


def initialize_state(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    lmda_major_0, lmda_minor_0, lmda_major_init, lmda_minor_init = derived_lengths(cfg)
    area = math.pi * lmda_major_0 * lmda_minor_0
    N = int(cfg.phi * cfg.L * cfg.L / area)

    x = rng.uniform(-cfg.L / 2.0, cfg.L / 2.0, N).astype(np.float64)
    y = rng.uniform(-cfg.L / 2.0, cfg.L / 2.0, N).astype(np.float64)
    theta = rng.uniform(0.0, 2.0 * math.pi, N).astype(np.float64)

    lmda_major = np.full(N, lmda_major_init, dtype=np.float64)
    lmda_minor = np.full(N, lmda_minor_init, dtype=np.float64)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    Lmda = np.empty((2, 2, N), dtype=np.float64)
    Lmda[0, 0, :] = cos_t * cos_t * lmda_major + sin_t * sin_t * lmda_minor
    Lmda[1, 1, :] = sin_t * sin_t * lmda_major + cos_t * cos_t * lmda_minor
    Lmda[0, 1, :] = sin_t * cos_t * (lmda_major - lmda_minor)
    Lmda[1, 0, :] = Lmda[0, 1, :]

    return rng, N, x, y, theta, Lmda, lmda_major, lmda_minor


def allocate_work_arrays(N: int):
    Fx = np.empty(N, dtype=np.float64)
    Fy = np.empty(N, dtype=np.float64)
    torque = np.empty(N, dtype=np.float64)
    sxx = np.empty(N, dtype=np.float64)
    sxy = np.empty(N, dtype=np.float64)
    syx = np.empty(N, dtype=np.float64)
    syy = np.empty(N, dtype=np.float64)
    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    return Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta


def choose_cell_count(L: float, cutoff: float) -> int:
    # int(L/cutoff) gives cell_size >= cutoff. At least 3 cells for stable 3x3 neighbour scan.
    return max(3, int(L / cutoff))


def compute_forces_cpu(force_kind: str, cfg: SimConfig, x, y, theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cutoff, Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, ):
    
    np.cos(theta, out=cos_theta)
    np.sin(theta, out=sin_theta)
    build_cell_list_cpu(x, y, cfg.L, n_cells, cell_head, cell_next)

    if force_kind == "morse":
        compute_forces_morse_cell_cpu(x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cfg.L, cutoff, cfg.epsilon, cfg.a_mor, cfg.D_mor, cfg.sigma_0, cfg.R_0, Fx, Fy,
            torque, sxx, sxy, syx, syy, )
            
    elif force_kind == "harmonic":
        compute_forces_harmonic_cell_cpu(x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cfg.L, cutoff, cfg.epsilon, cfg.beta, cfg.sigma_0, cfg.R_0, 
            Fx, Fy, torque, sxx, sxy, syx, syy,)
    else:
        raise ValueError(f"Unknown force_kind={force_kind!r}")


def save_snapshot(save_idx, x, y, theta, lmda_major, lmda_minor, Fx, Fy, sxx, sxy, syx, syy, traj):
    traj_x, traj_y, traj_theta, traj_lmaj, traj_lmin, traj_Fx, traj_Fy, traj_stress = traj
    traj_x[:, save_idx] = x
    traj_y[:, save_idx] = y
    traj_theta[:, save_idx] = theta
    traj_lmaj[:, save_idx] = lmda_major
    traj_lmin[:, save_idx] = lmda_minor
    traj_Fx[:, save_idx] = Fx
    traj_Fy[:, save_idx] = Fy
    traj_stress[0, 0, :, save_idx] = sxx
    traj_stress[0, 1, :, save_idx] = sxy
    traj_stress[1, 0, :, save_idx] = syx
    traj_stress[1, 1, :, save_idx] = syy


def run_simulation(cfg: SimConfig = CONFIG):
    set_num_threads(cfg.num_threads)

    steps = int(cfg.tot_time / cfg.dt)
    if steps <= 0:
        raise ValueError("tot_time/dt must give at least one step")
    if cfg.interval <= 0:
        raise ValueError("interval must be positive")

    lmda_major_0, lmda_minor_0, _, _ = derived_lengths(cfg)
    rng, N, x, y, theta, Lmda, lmda_major, lmda_minor = initialize_state(cfg)
    print(f"number of particles: {N}")

    cutoff = cfg.neighbour_cutoff_factor * cfg.sigma_0
    n_cells = choose_cell_count(cfg.L, cutoff)
    cell_size = cfg.L / n_cells
    
    if cell_size < cutoff:
        raise RuntimeError("cell_size must be >= cutoff for the 3x3 neighbour-cell scan")
            
    print(f"CPU cell list: {n_cells} x {n_cells} cells; cell_size={cell_size:.6g}; cutoff={cutoff:.6g}")

    cell_head = np.empty(n_cells * n_cells, dtype=np.int64)
    cell_next = np.empty(N, dtype=np.int64)

    Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta = allocate_work_arrays(N)

    n_save = max(1, steps // cfg.interval)
    traj_x = np.zeros((N, n_save), dtype=np.float64)
    traj_y = np.zeros((N, n_save), dtype=np.float64)
    traj_theta = np.zeros((N, n_save), dtype=np.float64)
    traj_lmaj = np.zeros((N, n_save), dtype=np.float64)
    traj_lmin = np.zeros((N, n_save), dtype=np.float64)
    traj_Fx = np.empty((N, n_save), dtype=np.float64)
    traj_Fy = np.empty((N, n_save), dtype=np.float64)
    traj_stress = np.zeros((2, 2, N, n_save), dtype=np.float64)
    traj = (traj_x, traj_y, traj_theta, traj_lmaj, traj_lmin, traj_Fx, traj_Fy, traj_stress)

    t0 = time.perf_counter()

    # Harmonic relaxation stage: same CPU linked-cell force path, with harmonic interaction.
    for _ in _progress(range(cfg.relax_steps), "relax"):
        compute_forces_cpu("harmonic", cfg, x, y, theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cutoff, Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, )
        noise_x = rng.normal(0.0, 1.0, N)
        noise_y = rng.normal(0.0, 1.0, N)
        noise_theta = rng.normal(0.0, 1.0, N)
        integrate_state_cpu(x, y, theta, Lmda, lmda_major, lmda_minor, Fx, Fy, torque, sxx, sxy, syx, syy, noise_x, noise_y, noise_theta, lmda_major_0, lmda_minor_0, cfg.deformable_relax,
            cfg.dt_relax, cfg.L, cfg.v0, cfg.D, cfg.D_r, cfg.R_0, cfg.tau, cfg.mu, cfg.K, cfg.incom_K, cfg.shape_diff_tol,)

    save_idx = 0
    for t in _progress(range(steps), "run"):
        compute_forces_cpu("harmonic", cfg, x, y, theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cutoff, Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, )

        noise_x = rng.normal(0.0, 1.0, N)
        noise_y = rng.normal(0.0, 1.0, N)
        noise_theta = rng.normal(0.0, 1.0, N)
        integrate_state_cpu(x, y, theta, Lmda, lmda_major, lmda_minor, Fx, Fy, torque, sxx, sxy, syx, syy, noise_x, noise_y, noise_theta, lmda_major_0, lmda_minor_0, 
        cfg.deformable, cfg.dt, cfg.L, cfg.v0, cfg.D, cfg.D_r, cfg.R_0, cfg.tau, cfg.mu, cfg.K, cfg.incom_K, cfg.shape_diff_tol, )

        if t % cfg.interval == 0 and save_idx < n_save:
            save_snapshot(save_idx, x, y, theta, lmda_major, lmda_minor, Fx, Fy, sxx, sxy, syx, syy, traj)
            save_idx += 1

    elapsed = time.perf_counter() - t0
    print(f"simulation finished in {elapsed:.3f} s using CPU linked-cell + prange")

    data_ellipses = {
        "meta": {
            **asdict(cfg),
            "N": N,
            "n_cells": n_cells,
            "cell_size": cell_size,
            "cutoff": cutoff,
            "sim_time": cfg.tot_time,
            "dt": cfg.dt,
            "actual_saved_frames": save_idx,},
        "x": traj_x[:, :save_idx],
        "y": traj_y[:, :save_idx],
        "orient": traj_theta[:, :save_idx],
        "l_major": traj_lmaj[:, :save_idx],
        "l_minor": traj_lmin[:, :save_idx],
        "Fx": traj_Fx[:, :save_idx],
        "Fy": traj_Fy[:, :save_idx],
        "stresses": traj_stress[:, :, :, :save_idx],}

    stem = f"{cfg.output_prefix}_dt_{cfg.dt}_phi_{cfg.phi:.2f}_v0_{cfg.v0:.2f}_tau_{cfg.tau:.2f}_mu_{cfg.mu:.2f}"
    if cfg.save_data:
        out_npy = Path(f"{stem}.npy")
        np.save(out_npy, data_ellipses, allow_pickle=True)
        print(f"saved data: {out_npy}")
    
    if cfg.save_hdf5:
        out_hdf = Path(f"{stem}.h5")
        save_hdf5_output(out_hdf, data_ellipses, cfg)
        print(f"saved HDF5 data: {out_hdf}")


    if cfg.make_movie and save_idx > 0:
        out_mp4 = Path(f"{stem}.mp4")
        make_movie(data_ellipses, cfg, out_mp4)
        print(f"saved movie: {out_mp4}")

    return data_ellipses


def save_hdf5_output(out_path, data_ellipse, cfg:SimConfig):
    '''
    Save trajectory output in HDF5 format.

    HDF5 layout:

    /meta/<attribute>          simulation metadata stored as HDF5 attributes
    /particles/x               shape (N, n_frames)
    /particles/y               shape (N, n_frames)
    /particles/orient          shape (N, n_frames)
    /particles/l_major         shape (N, n_frames)
    /particles/l_minor         shape (N, n_frames)
    /forces/Fx                 shape (N, n_frames)
    /forces/Fy                 shape (N, n_frames)
    /stresses/tensor           shape (2, 2, N, n_frames)
    /time/frame                saved frame index: 0, interval, 2*interval, ...
    /time/t                    physical saved times
    '''
    
    if h5py is None:
        raise ImportError("save_hdf5=True requires h5py")
    
    meta_data = data_ellipse["meta"]
    n_frames = data_ellipse["x"].shape[1]
    saved_frames = np.arange(n_frames, dtype=np.int64) * int(cfg.interval)
    saved_times = saved_frames.astype(np.float64) * float(cfg.dt)
    
    with h5py.File(out_path, "w") as h5:
        g_meta = h5.create_group("meta")
        
        for key,value in meta_data.items():
            g_meta.attrs[key] = value
    
        g_particles = h5.create_group("particles")
        g_forces = h5.create_group("forces")
        g_stresses = h5.create_group("stresses")
        g_time = h5.create_group("time")
        
        g_particles.create_dataset("x", data_ellipse["x"])
        g_particles.create_dataset("x", data_ellipse["y"])
        g_particles.create_dataset("orient", data_ellipse["orient"])
        g_particles.create_dataset("l_minor", data_ellipse["l_minor"])
        g_particles.create_dataset("l_major", data_ellipse["l_major"])

        g_forces.create_dataset("Fx", data_ellipse["Fx"])
        g_forces.create_dataset("Fy", data_ellipse["Fy"])

        g_stresses.create_dataset("stresses", data_ellipse["stresses"])

        g_time.create_dataset("frame", data=saved_frames)
        g_time.create_dataset("t", data=saved_times)

        h5["x"] = g_particles["x"]
        h5["y"] = g_particles["y"]
        h5["orient"] = g_particles["orient"]
        h5["l_major"] = g_particles["l_major"]
        h5["l_minor"] = g_particles["l_minor"]
        h5["Fx"] = g_forces["Fx"]
        h5["Fy"] = g_forces["Fy"]
        h5["stress_tensor"] = g_stresses["stresses"]


def load_hdf5_output(path):
    #Load the HDF5 output back into the same dictionary style as the .npy output.
    if h5py is None:
        raise ImportError("Loading HDF5 output requires h5py")

    with h5py.File(path, "r") as h5:
        meta = dict(h5["meta"].attrs)
        data = {
            "meta": meta,
            "x": h5["particles/x"][...],
            "y": h5["particles/y"][...],
            "orient": h5["particles/orient"][...],
            "l_major": h5["particles/l_major"][...],
            "l_minor": h5["particles/l_minor"][...],
            "Fx": h5["forces/Fx"][...],
            "Fy": h5["forces/Fy"][...],
            "stresses": h5["stresses/tensor"][...],
            "time": h5["time/t"][...],
            "frame": h5["time/frame"][...],
        }
    return data


def make_movie(data_ellipses, cfg: SimConfig, out_mp4: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import EllipseCollection

    traj_x = data_ellipses["x"]
    traj_y = data_ellipses["y"]
    traj_theta = data_ellipses["orient"]
    traj_lmda_major = data_ellipses["l_major"]
    traj_lmda_minor = data_ellipses["l_minor"]
    n_frames = traj_x.shape[1]

    deform_ratio = traj_lmda_major / traj_lmda_minor
    norm = plt.Normalize(1.0, 1.5)
    cmap = LinearSegmentedColormap.from_list("CustomCmap", ["green", "yellow", "red"])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(-cfg.L / 2.0, cfg.L / 2.0)
    ax.set_ylim(-cfg.L / 2.0, cfg.L / 2.0)
    ax.set_xticks([])
    ax.set_yticks([])

    def plot(frame):
        ax.clear()
        ax.set_xlim(-cfg.L / 2.0, cfg.L / 2.0)
        ax.set_ylim(-cfg.L / 2.0, cfg.L / 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ec = EllipseCollection( traj_lmda_major[:, frame] * 2.0, traj_lmda_minor[:, frame] * 2.0, traj_theta[:, frame] / math.pi * 180.0,
            units="x", offsets=np.array([traj_x[:, frame], traj_y[:, frame]]).T, offset_transform=ax.transData, array=deform_ratio[:, frame], cmap=cmap, norm=norm, )
        ax.add_collection(ec)
        ax.quiver(traj_x[:, frame], traj_y[:, frame], np.cos(traj_theta[:, frame]), np.sin(traj_theta[:, frame]), color="black",)
        ax.text(0.02, 0.97, f"frame {frame}", transform=ax.transAxes, va="top")
        return (ax,)

    animation = FuncAnimation(fig, plot, frames=n_frames, interval=500, blit=False, repeat=False)
    writer = FFMpegWriter(fps=cfg.movie_fps, codec="libx264", extra_args=["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p"],)
    animation.save(out_mp4, writer=writer, dpi=cfg.movie_dpi)
    plt.close(fig)


if __name__ == "__main__":
    run_simulation(CONFIG)


"""
param_template = Template(ellipse_template)

params = {
    "num_threads":7,
    "phi": 0.85,
    "dt": 0.001,
    "sim_time": 1000,
    "boxl": 40,
    "act_vel": 0.00,
    "Diff_t": 0.01,
    "Diff_r": 0.01,
    "a_morse": 2.5,
    "D_morse": 0.8,
    "gb_epsilon": 1.0,
    "beta_exp": 2,
    "R_0": 1.0,
    "sigma_0": 2.0,
    "aspect_ratio": 1.0,
    "aspect_ratio_init": 1.0,
    "KV_tau": 10.0,
    "KV_mu": 2.0,
    "KV_K": 50.0,
    "incom_K": True,
    "deform_flag": True,
    "deform_relax_flag":False,
    "dt_relax": 0.01,
    "relax_steps": int(5e4),
    "seed": 270526,
    "shape_diff_tol": 1e-5,
    "neighbour_list_cutoff":3.0,
    "save_interval": 200,
    "movie_name": "movie",
    "save_data": True,
    "save_h5":False,
    "make_movie":True,
    "movie_fps":25,
    "movie_dpi":120,
}

seeds = [1258]
act_vels = [0.0] #[0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
tau = 10.0
#mus = [0.1, 0.3, 0.5, 0.7, 1.0]
mus = [2.0]

phis = [0.85, 0.90, 0.95]
boxl = params["boxl"]
aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]

cur_dir = Path.cwd()
#ellipse_code_path = f"/data1/pabshettiwar/Simulations/Ellipsoids/Automate_python_scheduler/Ellipsoids_morse.py"
software_path = "python3"
print(cur_dir)

for asr in aspect_ratios:
    for v0 in act_vels:
        for phi in phis:
            for mu in mus:
                    seed = seeds[0]
                    
                    if not os.path.isdir(os.path.join(cur_dir, f"Seed-{seed}")):
                        os.mkdir(os.path.join(cur_dir, f"Seed-{seed}"))
                        path = os.path.join(cur_dir, f"Seed-{seed}")
                    else:
                        path = os.path.join(cur_dir, f"Seed-{seed}")

                    if not os.path.isdir(os.path.join(path, f"phi_{phi}_tau_{tau}_mu_{mu}_asr_{asr}") ):
                        os.mkdir(os.path.join(path, f"phi_{phi}_tau_{tau}_mu_{mu}_asr_{asr}"))
                        path = os.path.join(path,f"phi_{phi}_tau_{tau}_mu_{mu}_asr_{asr}")
                    else:
                        path = os.path.join(path,f"phi_{phi}_tau_{tau}_mu_{mu}_asr_{asr}")

                    print("current_path=", path)

                    params["seed"] = seed
                    params["aspect_ratio"] = asr
                    params["aspect_ratio_init"] = asr
                    params["act_vel"] = v0
                    params["KV_tau"] = tau
                    params["KV_mu"] = mu
                    params["phi"] = phi
                    params["movie_name"] = f"phi_{phi}_tau_{tau}_mu_{mu}_asr_{asr}"

                    #shutil.copy2(input_file_path,path)
                    #print(f"copied to the above path")

                    with open(os.path.join(path,f"ellipse_morse.py"), "w") as config_file:
                        config_data = param_template.render(params)
                        config_file.write(config_data)

                    curr_file_path = os.path.join(path,f"ellipse_morse.py")

                    # check for the cpu usage to fall below threshold
                    wait_for_cpu_usage_below(threshold = 60, check_interval=30)

                    # Run the job in the background
                    run_in_background(software_path, curr_file_path, path)

                    time.sleep(20)

'''
for seed in seeds:
    for v0 in act_vels:
        for tau in taus:
            for mu in mus:
                    
                    if not os.path.isdir(os.path.join(cur_dir, f"Seed-{seed}")):
                        os.mkdir(os.path.join(cur_dir, f"Seed-{seed}"))
                        path = os.path.join(cur_dir, f"Seed-{seed}")
                    else:
                        path = os.path.join(cur_dir, f"Seed-{seed}")

                    if not os.path.isdir(os.path.join(path, f"phi_{phi}_v0_{v0}_tau_{tau}_mu_{mu}") ):
                        os.mkdir(os.path.join(path, f"phi_{phi}_v0_{v0}_tau_{tau}_mu_{mu}"))
                        path = os.path.join(path, f"phi_{phi}_v0_{v0}_tau_{tau}_mu_{mu}")
                    else:
                        path = os.path.join(path,f"phi_{phi}_v0_{v0}_tau_{tau}_mu_{mu}")

                    print("current_path=", path)

                    params["seed"] = seed
                    params["act_vel"] = v0
                    params["KV_tau"] = tau
                    params["KV_mu"] = mu
                    params["movie_name"] = f"morse_L_{boxl}_phi_{phi:.2f}_v0_{v0}_tau_{tau}_mu_{mu}"

                    #shutil.copy2(input_file_path,path)
                    #print(f"copied to the above path")

                    with open(os.path.join(path,f"ellipse_morse.py"), "w") as config_file:
                        config_data = param_template.render(params)
                        config_file.write(config_data)

                    curr_file_path = os.path.join(path,f"ellipse_morse.py")

                    # check for the cpu usage to fall below threshold
                    wait_for_cpu_usage_below(threshold = 75, check_interval=80)

                    # Run the job in the background
                    run_in_background(software_path, curr_file_path, path)

                    time.sleep(30)



for fmin in f_min_morse:
    for dr in Dr_a:
        for J_align in J_align_a:
            for abp_act in abp_p_a:
                for seed in seeds:

                    D = round(8* fmin**2,4)
                    a = round(0.25*(1/fmin),4)

                    params["morse_a"] = a
                    params["morse_D"] = D
                    params["pair_nematic_J"] = J_align
                    params["abp_actreact_p"] = abp_act
                    params["Dr"] = dr
                    params["abp_type_1_v0"] = params["abp_actreact_p"]

                    if os.path.isdir(os.path.join(cur_dir, f"Seed-{seed}/xi_{xi}_J_{J_align}_dr_{dr}_abp-p_{abp_act}_ma_{a:.2f}_mD_{D:.2f}") ):
                        curr_path = os.path.join(cur_dir, f"Seed-{seed}/xi_{xi}_J_{J_align}_dr_{dr}_abp-p_{abp_act}_ma_{a:.2f}_mD_{D:.2f}")

                        # executable file - configuration file for samos
                        con_file_path = os.path.join(curr_path,f"tumoroid-pair_nematic.conf")

                        # check for the cpu usage to fall below threshold
                        wait_for_cpu_usage_below(threshold = 70, check_interval=20)
                        
                        # Run the job in the background
                        run_in_background(samos_abp_path, con_file_path, curr_path)

'''



