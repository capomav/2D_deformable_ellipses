from __future__ import annotations

#from dataclasses import dataclass, asdict
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



#@dataclass
class SimConfig:

     # Can be either "morse" or "harmonic"
    def __init__(self, interaction: str = "morse", deformable: bool = True, aspect_ratio: float = 1.0, KT: float = 0.0 ):

        self.interaction = interaction
        self.deformable = deformable 

        self.aspect_ratio = aspect_ratio
        self.aspect_ratio_init: float = self.aspect_ratio

        # Morse parameters
        self.a_mor: float = 2.5
        self.D_mor: float = 0.08
        
        # Harmonic potential parameters
        self.epsilon: float = 1.0
        self.beta: float = 2.0

        # deformable base radii
        self.R_0: float = 1.0
        self.sigma_0: float = 2.0

        self.deformable_relax: bool = False
        self.dt_relax: float = 0.01
        self.relax_steps: int = 10000

        # Cell-list neighbour cutoff
        self.neighbour_cutoff_factor: float = 3.0

        self.shape_diff_tol = 1.0e-5

        # friction and FDT
        self.zeta_t: float = 1.0
        self.zeta_r: float = 4*self.R_0**2/3 
        self.zeta_stress: float = 1.0
        self.KT = KT
        # if it's a thermal system
        if self.KT >0:
            self.D: float = self.KT/self.zeta_t
            self.D_r: float = self.KT/self.zeta_r
            self.D_str = self.KT/self.zeta_stress
        else:
            self.D = 0.0
            #self.Dr = 0.01
            self.D_str = 0.0

    # setting non-default parameters
    def set_syspar(self,phi: float = 0.6,  L: float = 20):
        self.phi=phi
        self.L = L

    def set_runpar(self, num_threads: int = 8, dt: float = 0.001, tot_time: float = 100, interval: int = 2000, seed: int = 17464 ):
        self.num_threads = num_threads
        self.interval = interval
        self.dt = dt
        self.tot_time = tot_time
        self.seed = seed	
        print('Setting simulation to run ')
		
    def set_actpar(self,v0: float = 0.3, D_r: float = 0.01):
        self.v0 = v0
        self.D_r = D_r

    # Deformable-particle parameters
    def set_defpar(self, tau: float = 10.0, mu: float = 0.5, K: float = 3.0, incom_K:bool = False):
        self.tau = tau
        self.mu = mu
        self.K = K
        self.incom_K = incom_K


## Small scalar helpers used by CPU Numba kernels

@njit(inline="always")
def pbc_delta(dx: float, L: float) -> float:
    """Minimum-image displacement for a square periodic box."""
    return dx - L * np.rint(dx / L)

@njit(inline="always")
def wrap_position(x: float, L: float) -> float:
    """Wrap coordinate to approximately [-L/2, L/2]."""
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
    """Build linked-cell list in O(N)."""
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
    #sigma = sigma_0 * sigma_arg ** (-0.5)
    
    sigma = sigma_0 * sigma_arg**(-0.5)* ((1+chi) / (1-chi))**(-0.25)

    Fx = 0.0
    Fy = 0.0
    torque = 0.0
    U_pot = 0.0

    if r < 2.0 * sigma:
        exp_term = math.exp(-a_mor * (r - sigma))
        common_force = - 2.0 * a_mor * D_mor * epsilon * exp_term * (1.0 - exp_term)
        common_shape = dots_add * dots_add / chi_add + dots_sub * dots_sub / chi_sub
        
        # adding a prefactor of half as we calculate energy for each pair twice in each particle loop
        U_pot = 0.5* D_mor * (1 - math.exp(-a_mor*(r - sigma)))**2 - 0.5 * D_mor

        geom_x = (rx / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2 * ((1+chi) / (1-chi))**(-0.5) ) * (-rx / (r2*r2) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1x + u2x) + dots_sub / chi_sub * (u1x - u2x)) ))
        geom_y = (ry / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2 * ((1+chi) / (1-chi))**(-0.5) ) * (-ry / (r2*r2) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1y + u2y) + dots_sub / chi_sub * (u1y - u2y)) ))

        Fx = common_force * geom_x
        Fy = common_force * geom_y

        common_du = - ( 0.5 * a_mor * D_mor * epsilon * chi * sigma_0 * ((1+chi) / (1-chi))**(-0.25) * exp_term * (1.0 - exp_term) * sigma_arg ** (-1.5) )

        du_geom_x = ( 2.0 * dots_add * rx / r2 / chi_add - dots_add * dots_add * chi * u2x / (chi_add ** 2)/r2 + 2.0 * dots_sub * rx / r2 / chi_sub + dots_sub * dots_sub * chi * u2x / (chi_sub ** 2)/r2 )
        du_geom_y = ( 2.0 * dots_add * ry / r2 / chi_add - dots_add * dots_add * chi * u2y / (chi_add ** 2)/r2 + 2.0 * dots_sub * ry / r2 / chi_sub + dots_sub * dots_sub * chi * u2y / (chi_sub ** 2)/r2 )

        dUdux = common_du * du_geom_x
        dUduy = common_du * du_geom_y
        torque = -u1x * dUduy + u1y * dUdux

    return Fx, Fy, torque, U_pot


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
    #sigma = sigma_0 * sigma_arg ** (-0.5)
    #New sigma definition 
    sigma = sigma_0 * sigma_arg**(-0.5)* ((1+chi) / (1-chi))**(-0.25)

    Fx = 0.0
    Fy = 0.0
    torque = 0.0
    U_pot = 0.0

    if r < sigma:
        common_force = - epsilon * (r - sigma) ** (beta - 1.0)
        common_shape = dots_add * dots_add / chi_add + dots_sub * dots_sub / chi_sub
        
        # adding a prefactor of half as we calculate energy for each pair twice in each particle loop
        U_pot = 0.5 * 1/beta * epsilon * (r - sigma)**beta

        geom_x = (rx / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2 * ((1+chi) / (1-chi))**(-0.5) ) * (-rx / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1x + u2x) + dots_sub / chi_sub * (u1x - u2x)) ))
        geom_y = (ry / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2 * ((1+chi) / (1-chi))**(-0.5) ) * (-ry / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1y + u2y) + dots_sub / chi_sub * (u1y - u2y)) ))
        
        # original
        #geom_x = (rx / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2) * (-rx / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1x + u2x) + dots_sub / chi_sub * (u1x - u2x)) ))
        #geom_y = (ry / r - (chi * sigma ** 3) / (2.0 * sigma_0 ** 2) * (-ry / (r ** 4) * common_shape + 1.0 / r2 * (dots_add / chi_add * (u1y + u2y) + dots_sub / chi_sub * (u1y - u2y)) ))

        Fx = common_force * geom_x
        Fy = common_force * geom_y

        common_du = - (0.25 * chi * sigma_0 * ((1+chi) / (1-chi))**(-0.25) * epsilon * (r - sigma) ** (beta - 1.0) * sigma_arg ** (-1.5))
        du_geom_x = (2.0 * dots_add * rx / r2 / chi_add - dots_add * dots_add * chi * u2x / (chi_add ** 2)/r2 + 2.0 * dots_sub * rx / r2 / chi_sub + dots_sub * dots_sub * chi * u2x / (chi_sub ** 2)/r2 )
        du_geom_y = (2.0 * dots_add * ry / r2 / chi_add - dots_add * dots_add * chi * u2y / (chi_add ** 2)/r2 + 2.0 * dots_sub * ry / r2 / chi_sub + dots_sub * dots_sub * chi * u2y / (chi_sub ** 2)/r2 )

        dUdux = common_du * du_geom_x
        dUduy = common_du * du_geom_y
        torque = - u1x * dUduy + u1y * dUdux

    return Fx, Fy, torque, U_pot



@njit(parallel=True)
def compute_forces_morse_cell_cpu(x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells: int, L: float, cutoff: float, epsilon: float, a_mor: float, D_mor: float,
    sigma_0: float, R_0: float, Fx, Fy, torque_mech, sxx, sxy, syx, syy, U_potential, ):
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
        u_pot_i = 0.0

        for dcx in range(-1, 2):
            cx = (cxi + dcx) % n_cells
            for dcy in range(-1, 2):
                cy = (cyi + dcy) % n_cells
                cid = cx + n_cells * cy
                j = cell_head[cid]
                while j != -1:
                    if j != i:
                        #dx = pbc_delta(x[j] - xi, L)
                        dx = pbc_delta(xi - x[j], L)
                        #dy = pbc_delta(y[j] - yi, L)
                        dy = pbc_delta(yi - y[j], L)
                        r2 = dx * dx + dy * dy
                        if 1.0e-24 < r2 < cutoff2:
                            r = math.sqrt(r2)
                            sigma_e_corrected = lmda_major[i] + lmda_major[j]
                            sigma_s_corrected = lmda_minor[i] + lmda_minor[j]
                            ar2 = (sigma_e_corrected / sigma_s_corrected) ** 2
                            chi_corrected = (ar2 - 1.0) / (ar2 + 1.0)

                            fxj, fyj, torque_j, u_pot = calculate_force_torque_morse_scalar(dx, dy, r, chi_corrected, u1x, u1y, cos_theta[j], sin_theta[j], epsilon, a_mor, D_mor, sigma_0, )
                            fxi += fxj
                            fyi += fyj
                            ti += torque_j
                            sxx_i += dx * fxj
                            sxy_i += dx * fyj
                            syx_i += dy * fxj
                            syy_i += dy * fyj
                            u_pot_i+= u_pot
                    j = cell_next[j]

        Fx[i] = fxi
        Fy[i] = fyi
        torque_mech[i] = ti
        sxx[i] = stress_prefactor * sxx_i
        sxy[i] = stress_prefactor * sxy_i
        syx[i] = stress_prefactor * syx_i
        syy[i] = stress_prefactor * syy_i
        U_potential[i] = u_pot_i


@njit(parallel=True)
def compute_forces_harmonic_cell_cpu(x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells: int, L: float, cutoff: float, epsilon: float, beta: float, sigma_0: float, R_0: float, Fx, Fy, torque_mech, sxx, sxy, syx, syy, U_potential, ):
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
        u_pot_i = 0.0

        for dcx in range(-1, 2):
            cx = (cxi + dcx) % n_cells
            for dcy in range(-1, 2):
                cy = (cyi + dcy) % n_cells
                cid = cx + n_cells * cy
                j = cell_head[cid]
                while j != -1:
                    if j != i:
                        #dx = pbc_delta(x[j] - xi, L)
                        dx = pbc_delta(xi - x[j], L)
                        #dy = pbc_delta(y[j] - yi, L)
                        dy = pbc_delta(yi - y[j], L)
                        r2 = dx * dx + dy * dy
                        if 1.0e-24 < r2 < cutoff2:
                            r = math.sqrt(r2)
                            sigma_e_corrected = lmda_major[i] + lmda_major[j]
                            sigma_s_corrected = lmda_minor[i] + lmda_minor[j]
                            ar2 = (sigma_e_corrected / sigma_s_corrected) ** 2
                            chi_corrected = (ar2 - 1.0) / (ar2 + 1.0)
                            

                            fxj, fyj, torque_j, u_pot = calculate_force_torque_harmonic_scalar(dx, dy, r, chi_corrected, u1x, u1y, cos_theta[j], sin_theta[j], epsilon, beta, sigma_0, )
                            fxi += fxj
                            fyi += fyj
                            ti += torque_j
                            sxx_i += dx * fxj
                            sxy_i += dx * fyj
                            syx_i += dy * fxj
                            syy_i += dy * fyj
                            u_pot_i+= u_pot
                    j = cell_next[j]

        Fx[i] = fxi
        Fy[i] = fyi
        torque_mech[i] = ti
        sxx[i] = stress_prefactor * sxx_i
        sxy[i] = stress_prefactor * sxy_i
        syx[i] = stress_prefactor * syx_i
        syy[i] = stress_prefactor * syy_i
        U_potential[i] = u_pot_i


# State update 

@njit(parallel=True)
def integrate_state_cpu(x, y, theta, Lmda, lmda_major, lmda_minor, Fx, Fy, torque_mech, sxx, sxy, syx, syy, noise_x, noise_y, noise_theta, noise_sxx, noise_syy, lmda_major_0: float, lmda_minor_0: float, deformable: bool,
    dt_step: float, L: float, v0: float, D: float, D_r: float, D_str:float, zeta_t:float, zeta_r:float, R_0: float, tau: float, mu: float, K:float, incom_K: bool, shape_diff_tol:float, ):
    
    n = x.shape[0]
    tol = shape_diff_tol #1.0e-5
    sqrt_2Ddt = math.sqrt(2.0 * D * dt_step)
    sqrt_2Drdt = math.sqrt(2.0 * D_r * dt_step)
    sqrt_2Dstrdt = math.sqrt(2.0*D_str* dt_step)
    
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
            mu_dynamic = mu * (1.0 + 0.0 * delta_lmda * delta_lmda)
            
            # Adding noise to stresses component wise
            
            sxx[i] += sqrt_2Dstrdt * noise_sxx[i]
            syy[i] += sqrt_2Dstrdt * noise_syy[i]

            #dL00 = -(L00 - L00_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (sxx[i] - syy[i])
            #dL11 = -(L11 - L11_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (syy[i] - sxx[i])
            #dL01 = -(L01 - L01_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * sxy[i]
            #dL10 = -(L10 - L10_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * syx[i]
            
            sxx[i] = - sxx[i]
            syy[i] = - syy[i]
            sxy[i] = - sxy[i]
            syx[i] = - syx[i]
            
            if(incom_K):

                dL00 = -(L00 - L00_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (sxx[i] - syy[i])
                dL11 = -(L11 - L11_0) / tau + R_0 / (4.0 * tau * mu_dynamic) * (syy[i] - sxx[i])
                dL01 = -(L01 - L01_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * sxy[i]
                dL10 = -(L10 - L10_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * syx[i]
            else:
                dL00 = -(L00 - L00_0) / tau + R_0 / (4.0 * tau * mu_dynamic * K) * ((mu_dynamic + K) * sxx[i] + (mu_dynamic - K) * syy[i])
                dL11 = -(L11 - L11_0) / tau + R_0 / (4.0 * tau * mu_dynamic * K) * ((mu_dynamic + K) * syy[i] + (mu_dynamic - K) * sxx[i])
                dL01 = -(L01 - L01_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * sxy[i]
                dL10 = -(L10 - L10_0) / tau + R_0 / (2.0 * tau * mu_dynamic) * syx[i]

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
            
            #if abs(Ldiff) < tol:
            #    if Ldiff >= 0.0:
            #        Ldiff = tol
            #    else:
            #        Ldiff = -tol

            root = math.sqrt(0.25 * Ldiff * Ldiff + L01 * L01)
            lmaj = 0.5 * (L00 + L11) + root
            lmin = 0.5 * (L00 + L11) - root

            theta_def = 0.5 * math.atan2(2.0 * L01, Ldiff)
            theta_def = theta_def - math.pi * np.rint((theta_def - th) / math.pi)

            th += (1/zeta_r)*torque_mech[i] * dt_step + (1/zeta_r)*(theta_def - th) + sqrt_2Drdt * noise_theta[i]
            th = wrap_position(th, 2.0 * math.pi)
        else:
            th += (1/zeta_r)*torque_mech[i] * dt_step + sqrt_2Drdt * noise_theta[i]
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

        x[i] = wrap_position(x[i] + (1/zeta_t)*(v0 * co + Fx[i]) * dt_step + sqrt_2Ddt * noise_x[i], L)
        y[i] = wrap_position(y[i] + (1/zeta_t)*(v0 * si + Fy[i]) * dt_step + sqrt_2Ddt * noise_y[i], L)


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
    U_potential = np.empty(N, dtype=np.float64)
    return Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, U_potential


def choose_cell_count(L: float, cutoff: float) -> int:
    # int(L/cutoff) gives cell_size >= cutoff. At least 3 cells for stable 3x3 neighbour scan.
    return max(3, int(L / cutoff))


def compute_forces_cpu(force_kind: str, cfg: SimConfig, x, y, theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cutoff, Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, U_potential ):
    
    np.cos(theta, out=cos_theta)
    np.sin(theta, out=sin_theta)
    build_cell_list_cpu(x, y, cfg.L, n_cells, cell_head, cell_next)

    if force_kind == "morse":
        compute_forces_morse_cell_cpu(x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cfg.L, cutoff, cfg.epsilon, cfg.a_mor, cfg.D_mor, cfg.sigma_0, cfg.R_0, Fx, Fy,
            torque, sxx, sxy, syx, syy, U_potential,)
            
    elif force_kind == "harmonic":
        
        compute_forces_harmonic_cell_cpu(x, y, cos_theta, sin_theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cfg.L, cutoff, cfg.epsilon, cfg.beta, cfg.sigma_0, cfg.R_0, 
            Fx, Fy, torque, sxx, sxy, syx, syy, U_potential,)
    else:
        raise ValueError(f"Unknown force_kind={force_kind!r}")


def save_snapshot(save_idx, x, y, theta, lmda_major, lmda_minor, Fx, Fy, sxx, sxy, syx, syy, traj, traj_u_potential, U_potential):
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
    traj_u_potential[:, save_idx] = U_potential


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
        g_energies = h5.create_group("energies")
        g_time = h5.create_group("time")
        
        g_particles.create_dataset("x", data_ellipse["x"])
        g_particles.create_dataset("y", data_ellipse["y"])
        g_particles.create_dataset("orient", data_ellipse["orient"])
        g_particles.create_dataset("l_minor", data_ellipse["l_minor"])
        g_particles.create_dataset("l_major", data_ellipse["l_major"])

        g_forces.create_dataset("Fx", data_ellipse["Fx"])
        g_forces.create_dataset("Fy", data_ellipse["Fy"])

        g_stresses.create_dataset("tensor", data_ellipse["stresses"])
        g_energies.create_dataset("u_potential", data_ellipse["u_potential"])
        g_energies.create_dataset("u_potential_total", data_ellipse["u_potential"].sum(axis=0))

        g_time.create_dataset("frame", data=saved_frames)
        g_time.create_dataset("t", data=saved_times)

        h5["x"] = g_particles["x"]
        h5["y"] = g_particles["y"]
        h5["orient"] = g_particles["orient"]
        h5["l_major"] = g_particles["l_major"]
        h5["l_minor"] = g_particles["l_minor"]
        h5["Fx"] = g_forces["Fx"]
        h5["Fy"] = g_forces["Fy"]
        h5["stress_tensor"] = g_stresses["tensor"]
        h5["u_potential"] = g_energies["u_potential"]
        h5["u_potential_total"] = g_energies["u_potential_total"]


def load_hdf5_output(path):
    """Load the HDF5 output back into the same dictionary style as the .npy output."""
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
            "u_potential": h5["energies/u_potential"][...],
            "u_potential_total": h5["energies/u_potential_total"][...],
            "time": h5["time/t"][...],
            "frame": h5["time/frame"][...],
        }
    return data
'''
    traj_x = data_ellipses["x"]
    traj_y = data_ellipses["y"]
    traj_theta = data_ellipses["orient"]
    traj_lmda_major = data_ellipses["l_major"]
    traj_lmda_minor = data_ellipses["l_minor"]
    n_frames = traj_x.shape[1]

    deform_ratio = traj_lmda_major / traj_lmda_minor
    norm = plt.Normalize(cfg.aspect_ratio-0.5, cfg.aspect_ratio+0.5)
    cmap = LinearSegmentedColormap.from_list("CustomCmap", ["green", "yellow", "red"])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(-cfg.L / 2.0, cfg.L / 2.0)
    ax.set_ylim(-cfg.L / 2.0, cfg.L / 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    def plot(frame):
        ax.clear()
        ax.set_xlim(-cfg.L / 2.0, cfg.L / 2.0)
        ax.set_ylim(-cfg.L / 2.0, cfg.L / 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ec = EllipseCollection( traj_lmda_major[:, frame] * 2.0, traj_lmda_minor[:, frame] * 2.0, traj_theta[:, frame] / math.pi * 180.0,
            units="xy", offsets=np.array([traj_x[:, frame], traj_y[:, frame]]).T, offset_transform=ax.transData, array=deform_ratio[:, frame], cmap=cmap, norm=norm, )
        ax.add_collection(ec)
        ax.quiver(traj_x[:, frame], traj_y[:, frame], np.cos(traj_theta[:, frame]), np.sin(traj_theta[:, frame]), color="black",)
        ax.text(0.02, 0.97, f"frame {frame}", transform=ax.transAxes, va="top")
        return (ax,)

    animation = FuncAnimation(fig, plot, frames=n_frames, interval=cfg.frame_interval, blit=False, repeat=False)
    writer = FFMpegWriter(fps=cfg.movie_fps, codec="libx264", extra_args=["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p"],)
    animation.save(out_mp4, writer=writer, dpi=cfg.movie_dpi)
    plt.close(fig)
'''



def make_movie(data_ellipses, cfg: SimConfig, out_mp4: Path, plot_energy = False, frame_interval = 2000, movie_fps = 25, movie_dpi = 100):
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
    traj_u_potential = data_ellipses["u_potential"]
    n_frames = traj_x.shape[1]

    if traj_u_potential.shape != traj_x.shape:
        raise ValueError(
            "u_potential must have shape (N, n_frames), matching the position trajectories"
        )

    # U_potential already contains the 1/2 pair-counting prefactor, so summing
    # over all particles gives the total potential energy of the system.
    total_u_potential = np.sum(traj_u_potential, axis=0)
    saved_times = np.arange(n_frames, dtype=np.float64) * cfg.interval * cfg.dt

    deform_ratio = traj_lmda_major / traj_lmda_minor
    norm = plt.Normalize(cfg.aspect_ratio - 0.2, cfg.aspect_ratio + 0.2)
    cmap = LinearSegmentedColormap.from_list(
        "CustomCmap", ["green", "yellow", "red"]
    )

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(-cfg.L / 2.0, cfg.L / 2.0)
    ax.set_ylim(-cfg.L / 2.0, cfg.L / 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    if plot_energy:
        # Inset in the upper-right corner. Coordinates are fractions of the figure.
        ax_energy = fig.add_axes([0.60, 0.68, 0.35, 0.25])
        energy_line, = ax_energy.plot([], [], linewidth=1.5)
        current_energy, = ax_energy.plot([], [], marker="o", markersize=3)
        ax_energy.set_title("Potential energy", fontsize=9)
        ax_energy.set_xlabel("time", fontsize=8)
        ax_energy.set_ylabel(r"$U_{\mathrm{pot}}$", fontsize=8)
        ax_energy.tick_params(axis="both", labelsize=7)
        ax_energy.grid(alpha=0.25)
        ax_energy.set_facecolor((1.0, 1.0, 1.0, 0.88))

        if n_frames == 1:
            ax_energy.set_xlim(-0.5 * cfg.interval * cfg.dt, 0.5 * cfg.interval * cfg.dt)
        else:
            ax_energy.set_xlim(saved_times[0], saved_times[-1])

        finite_energy = total_u_potential[np.isfinite(total_u_potential)]
        if finite_energy.size == 0:
            energy_min, energy_max = -1.0, 1.0
        else:
            energy_min = float(np.min(finite_energy))
            energy_max = float(np.max(finite_energy))
            if np.isclose(energy_min, energy_max):
                padding = max(1.0, 0.05 * abs(energy_min))
            else:
                padding = 0.05 * (energy_max - energy_min)
            energy_min -= padding
            energy_max += padding
        ax_energy.set_ylim(energy_min, energy_max)

    def plot(frame):
        ax.clear()
        ax.set_xlim(-cfg.L / 2.0, cfg.L / 2.0)
        ax.set_ylim(-cfg.L / 2.0, cfg.L / 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        ec = EllipseCollection(
            traj_lmda_major[:, frame] * 2.0,
            traj_lmda_minor[:, frame] * 2.0,
            traj_theta[:, frame] / math.pi * 180.0,
            units="xy",
            offsets=np.column_stack((traj_x[:, frame], traj_y[:, frame])),
            offset_transform=ax.transData,
            array=deform_ratio[:, frame],
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.add_collection(ec)
        ax.quiver(
            traj_x[:, frame],
            traj_y[:, frame],
            np.cos(traj_theta[:, frame]),
            np.sin(traj_theta[:, frame]),
            color="black",
        )

        current_time = saved_times[frame]
        ax.text(
            0.02,
            0.97,
            f"frame {frame}   t = {current_time:.4g}",
            transform=ax.transAxes,
            va="top",
        )

        if plot_energy:
            # Display only the energy history available up to this movie frame.
            energy_line.set_data(
                saved_times[: frame + 1],
                total_u_potential[: frame + 1],
            )
            current_energy.set_data(
                [current_time],
                [total_u_potential[frame]],
            )

            return ax, energy_line, current_energy
        else:
            return ax

    animation = FuncAnimation(
        fig,
        plot,
        frames=n_frames,
        interval=frame_interval,
        blit=False,
        repeat=False,
    )
    writer = FFMpegWriter(
        fps=movie_fps,
        codec="libx264",
        extra_args=["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p"],
    )
    animation.save(out_mp4, writer=writer, dpi=movie_dpi)
    plt.close(fig)

def run_simulation(cfg: SimConfig):
    set_num_threads(cfg.num_threads)

    steps = int(cfg.tot_time / cfg.dt)
    
    if steps <= 0:
        raise ValueError("tot_time/dt must give at least one step")
    if cfg.interval <= 0:
        raise ValueError("interval must be positive")

    # initialize target major and minor axisß
    lmda_major_0, lmda_minor_0, _, _ = derived_lengths(cfg)
    rng, N, x, y, theta, Lmda, lmda_major, lmda_minor = initialize_state(cfg)
    print(f"number of particles: {N}")

    # cell-list setup ----------------------------
    cutoff = cfg.neighbour_cutoff_factor * cfg.sigma_0
    n_cells = choose_cell_count(cfg.L, cutoff)
    cell_size = cfg.L / n_cells
    
    if cell_size < cutoff:
        raise RuntimeError("cell_size must be >= cutoff for the 3x3 neighbour-cell scan")
            
    print(f"CPU cell list: {n_cells} x {n_cells} cells; cell_size={cell_size:.6g}; cutoff={cutoff:.6g}")

    # cross-check


    cell_head = np.empty(n_cells * n_cells, dtype=np.int64)
    cell_next = np.empty(N, dtype=np.int64)
    # ----------------------------------------------------

    Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, U_potential = allocate_work_arrays(N)

    # sets up arrays to be saved in the end
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
    traj_u_potential = np.zeros((N, n_save), dtype=np.float64)

    t0 = time.perf_counter()

    # Harmonic relaxation stage: same CPU linked-cell force path, with harmonic interaction.
    # and no self-propulsion
    for _ in _progress(range(cfg.relax_steps), "relax"):
        compute_forces_cpu("harmonic", cfg, x, y, theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cutoff, Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, U_potential, )
        noise_x = rng.normal(0.0, 1.0, N)
        noise_y = rng.normal(0.0, 1.0, N)
        noise_sxx = rng.normal(0.0, 1.0, N)
        noise_syy = rng.normal(0.0, 1.0, N)
        noise_theta = rng.normal(0.0, 1.0, N)
        integrate_state_cpu(x, y, theta, Lmda, lmda_major, lmda_minor, Fx, Fy, torque, sxx, sxy, syx, syy, noise_x, noise_y, noise_theta, noise_sxx, noise_syy, lmda_major_0, lmda_minor_0, cfg.deformable_relax,
            cfg.dt_relax, cfg.L, 0.0, 0.0, 0.0, cfg.D_str, cfg.zeta_t, cfg.zeta_r, cfg.R_0, cfg.tau, cfg.mu, cfg.K, cfg.incom_K, cfg.shape_diff_tol,)

    save_idx = 0
    for t in _progress(range(steps), "run"):
        
        compute_forces_cpu(cfg.interaction, cfg, x, y, theta, lmda_major, lmda_minor, cell_head, cell_next, n_cells, cutoff, Fx, Fy, torque, sxx, sxy, syx, syy, cos_theta, sin_theta, U_potential,  )

        noise_x = rng.normal(0.0, 1.0, N)
        noise_y = rng.normal(0.0, 1.0, N)
        noise_theta = rng.normal(0.0, 1.0, N)
        integrate_state_cpu(x, y, theta, Lmda, lmda_major, lmda_minor, Fx, Fy, torque, sxx, sxy, syx, syy, noise_x, noise_y, noise_theta, noise_sxx, noise_syy, lmda_major_0, lmda_minor_0, 
        cfg.deformable, cfg.dt, cfg.L, cfg.v0, cfg.D, cfg.D_r, cfg.D_str, cfg.zeta_t, cfg.zeta_r, cfg.R_0, cfg.tau, cfg.mu,  cfg.K, cfg.incom_K, cfg.shape_diff_tol, )

        if t % cfg.interval == 0 and save_idx < n_save:
            save_snapshot(save_idx, x, y, theta, lmda_major, lmda_minor, Fx, Fy, sxx, sxy, syx, syy, traj, traj_u_potential, U_potential)
            save_idx += 1

    elapsed = time.perf_counter() - t0
    print(f"simulation finished in {elapsed:.3f} s using CPU linked-cell + prange")

    data_ellipses = {
        "meta": {
            #**asdict(cfg),
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
        "stresses": traj_stress[:, :, :, :save_idx],
        "u_potential": traj_u_potential[:, :save_idx],}
    
    return data_ellipses

    

    
# Essentially a sample script
if __name__ == "__main__":

    #### Set up the simulation along the following pattern:
    #### Not setting some of the arguments will use the default values
        
    # Construct new simulation
    # def __init__(self, interaction: str = "morse", deformable: bool = True, aspect_ratio: float = 1.0, KT: float = 0.0 ):
    cfg = SimConfig(interaction = 'morse', deformable = True, aspect_ratio = 1.2, KT = 0.0)
        
    ## Set its various parameters:
    # packing fraction and system size
    cfg.set_syspar(phi = 0.8, L=20)
    # activity
    cfg.set_actpar(v0 = 0.2, D_r = 0.01)
    # Deformable-particle parameters
    cfg.set_defpar(tau = 10.0, mu = 0.5 ,K = 1.0, incom_K = False)
    # simulation run
    cfg.set_runpar(num_threads = 8, dt = 0.001, tot_time = 100, interval = 1000, seed = 12345 )

    # Execute the actual simulation, return the data to be saved	
    data_ellipses = run_simulation(cfg)

    # Sanity check on output - in particular stop movie saving if it's garbage

    # Output controls
    save_data = True
    save_hdf5 = False
    save_movie = True
    output_prefix = "interaction"
    movie_fps = 25
    movie_dpi = 100
    frame_interval = 1000
        
    # Name of output file
    stem = f"{output_prefix}_{cfg.interaction}_phi_{cfg.phi:.2f}_v0_{cfg.v0:.2f}_tau_{cfg.tau:.2f}_mu_{cfg.mu:.2f}_K_{cfg.K:.2f}_asr_{cfg.aspect_ratio}_deform_{cfg.deformable}"

    if save_data:
        out_npy = Path(f"{stem}.npy")
        np.save(out_npy, data_ellipses, allow_pickle=True)
        print(f"saved data: {out_npy}")
                
    if save_hdf5:
        out_hdf = Path(f"{stem}.h5")
        save_hdf5_output(out_hdf, data_ellipses, cfg)
        print(f"saved HDF5 data: {out_hdf}")

    #def make_movie(data_ellipses, cfg: SimConfig, out_mp4: Path, plot_energy = False, frame_interval = 2000, movie_fps = 25, movie_dpi = 100):
    if save_movie:
        out_mp4 = Path(f"{stem}.mp4")
        make_movie(data_ellipses, cfg, out_mp4, False, frame_interval, movie_fps, movie_dpi)
        print(f"saved movie: {out_mp4}")
