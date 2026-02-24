
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
        usage = psutil.cpu_percent(interval=100)
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
        # Construct the command:
        #  - nohup to allow process to continue if user logs out
        #  - output/error redirected to nohup_output.log
        #  - & to run in background
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import EllipseCollection
import numpy as np
import time
from numba import jit, njit, prange, vectorize, set_num_threads
import timeit
import tqdm

# Set the total number of cores to be used(need to match with the job script)

set_num_threads({{num_threads}})
#constants
phi = {{phi}}              #packing fraction; 0.005 for 1 particle
dt = {{dt}}               #timestep
tot_time = {{sim_time}}
steps = int(tot_time/dt)
L = {{boxl}}                  #system size
v0 = {{act_vel}}               #active velocity

D = {{Diff_t}}               #diffusion constant; T/drag
D_r = {{Diff_r}}              #rotational diffusion constant; T/drag_r

epsilon = {{gb_epsilon}}           #energy (an)isotropy; constant for now means isotropic
beta = {{beta_exp}}                #power in the potential; 2 for harmonic spring
R_0 = {{R_0}}
sigma_0 = {{sigma_0}}
aspect_ratio = {{aspect_ratio}}      #ratio of major axis/minor axis; deformation only works if it is 1
aspect_ratio_init = {{aspect_ratio_init}}
sigma_edge = R_0*np.sqrt(aspect_ratio)
sigma_side = R_0/np.sqrt(aspect_ratio)
lmda_major_0 = sigma_edge      #rest lambda of major axis
lmda_minor_0 = sigma_side      #rest lambda of minor axis
lmda_major_init = lmda_major_0 #*aspect_ratio_init/2
lmda_minor_init = lmda_minor_0 #sigma_0/2


tau = {{KV_tau}}
mu = {{KV_mu}}
K = {{KV_K}}

# flag to switch on the deformations
deformable = {{deform_flag}}

dt_relax = {{dt_relax}}
relax_steps = {{relax_steps}}

np.random.seed({{seed}})

#initializing

area = np.pi*lmda_major_0*lmda_minor_0
N = int(phi*L**2/area)
x = np.random.uniform(-L/2, L/2, N)
y = np.random.uniform(-L/2, L/2, N)
theta = np.random.uniform(0, 2*np.pi, N)
print("number of particles : ",  N)


lmda_major = np.ones(N)*lmda_major_init
lmda_minor = np.ones(N)*lmda_minor_init
sin = np.sin(theta)
cos = np.cos(theta)
Lmda = np.array([[cos**2*lmda_major+sin**2*lmda_minor, sin*cos*(lmda_major-lmda_minor)], [sin*cos*(lmda_major-lmda_minor), sin**2*lmda_major+cos**2*lmda_minor]])


@vectorize()
def periodic_boundaries(x,L):
    return x - L*np.round(x/L)

@njit()
def force_12(epsilon, beta, r, sigma, chi, sigma_0, chi_add, chi_sub, dots_add, dots_sub, rxi, u1xi, u2xi):
    return 1/2*epsilon*beta*(r-sigma)**(beta-1) *(rxi/r - (chi*sigma**3)/(2*sigma_0**2) * (-rxi/r**4 * (dots_add**2/chi_add + dots_sub**2/chi_sub) + 1/r**2 * (dots_add/chi_add*(u1xi+ u2xi) + dots_sub/chi_sub*(u1xi - u2xi))))
    #return F_xi

@njit()
def dUduxi_12(epsilon, beta, r, sigma, sigma_0, chi, chi_add, chi_sub, dots_add, dots_sub, rxi, u1xi, u2xi):
    return 1/8*chi*sigma_0*epsilon*beta*(r-sigma)**(beta-1)* (1 - 0.5*chi/r**2 * (dots_add**2/chi_add + dots_sub**2/chi_sub))**(-3/2) * (2*dots_add*rxi/r/chi_add + dots_add**2*chi*u2xi/chi_add**2 + 2*dots_sub*rxi/r/chi_sub + dots_sub**2*chi*u2xi/chi_sub**2)
    #return dUduxi


@njit()
def calculate_force_torque(rx, ry, r, chi, u1, u2):

    #note: u are normalized, r are not
    rdotu1 = rx*u1[0] + ry*u1[1]
    rdotu2 = rx*u2[0] + ry*u2[1]
    u1dotu2 = u1[0]*u2[0] + u1[1]*u2[1]
    dots_add = rdotu1 + rdotu2
    dots_sub = rdotu1 - rdotu2
    # chi is corrected and passed in argument
    # chi = ((sigma_edge/sigma_side)**2-1) / ((sigma_edge/sigma_side)**2+1)
    chi_add = 1 + chi*u1dotu2
    chi_sub = 1 - chi*u1dotu2
    sigma = sigma_0*(1 - chi/2/r**2 * (dots_add**2/chi_add + dots_sub**2/chi_sub))**(-1/2)

    if r < sigma:

        Fx = force_12(epsilon, beta, r, sigma, chi, sigma_0, chi_add, chi_sub, dots_add, dots_sub, rx, u1[0], u2[0])
        Fy = force_12(epsilon, beta, r, sigma, chi, sigma_0, chi_add, chi_sub, dots_add, dots_sub, ry, u1[1], u2[1])

        dUdux = dUduxi_12(epsilon, beta, r, sigma, sigma_0, chi, chi_add, chi_sub, dots_add, dots_sub, rx, u1[0], u2[0])
        dUduy = dUduxi_12(epsilon, beta, r, sigma, sigma_0, chi, chi_add, chi_sub, dots_add, dots_sub, ry, u1[1], u2[1])

        torque = u1[0]*dUduy - u1[1]*dUdux

    else:

        Fx = 0
        Fy = 0
        torque = 0

    return Fx, Fy, torque

@njit()
def neighbour_forces_calc(dis_x,dis_y, lmda_major, lmda_minor, dis_abs, u_vecs, neighbours, i):

    fx = 0.0
    fy = 0.0
    rcrossF = np.zeros((2,2))
    torq = 0.0

    for j in neighbours:
        sigma_e_corrected = (lmda_major[i] + lmda_major[j])
        sigma_s_corrected = (lmda_minor[i] + lmda_minor[j])
        chi_corrected = ( (sigma_e_corrected/sigma_s_corrected)**2 - 1)/( (sigma_e_corrected/sigma_s_corrected)**2 +1 )

        Fx_j, Fy_j, torque_j = calculate_force_torque(dis_x[j], dis_y[j], dis_abs[j], chi_corrected, u_vecs[:,i], u_vecs[:,j])
        fx += Fx_j
        fy += Fy_j
        rcrossF += np.outer([dis_x[j], dis_y[j]], [Fx_j, Fy_j])
        torq += torque_j

    return fx,fy,rcrossF,torq


@njit(parallel=True)
def update(x,y, theta, Lmda, lmda_major, lmda_minor, lmda_major_0, lmda_minor_0, deformable, t): #t is there for debug reasons

    #calculating forces and stresses
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    stress= np.zeros((2,2,N))
    torque_mech = np.zeros(N)
    u_vecs = np.stack((np.cos(theta), np.sin(theta)))
    sin = np.sin(theta)
    cos = np.cos(theta)
    Lmda_0 = np.empty((2, 2, N), dtype=np.float64)

    Lmda_0[0, 0, :] = cos*cos*lmda_major_0 + sin*sin*lmda_minor_0
    Lmda_0[0, 1, :] = sin*cos*(lmda_major_0 - lmda_minor_0)
    Lmda_0[1, 0, :] = Lmda_0[0, 1, :]
    Lmda_0[1, 1, :] = sin*sin*lmda_major_0 + cos*cos*lmda_minor_0

    for i in prange(N):
        dis_x = periodic_boundaries(x - x[i], L)
        dis_y = periodic_boundaries(y - y[i], L)
        dis_abs = np.sqrt(dis_x**2 + dis_y**2 )
        neighbours = np.where(dis_abs<3*sigma_0)[0]
        neighbours = neighbours[neighbours!=i]
        fx,fy,rcrossF,torq = neighbour_forces_calc(dis_x,dis_y, lmda_major, lmda_minor, dis_abs, u_vecs,neighbours,i) # Correct : add lambda major and add lambda minor
        Fx[i]+= fx
        Fy[i]+= fy
        torque_mech[i]+= torq
        stress[:,:,i] += 1/(2*np.pi*R_0**2)*rcrossF


    if (deformable):

         #updating
         dLmda_dt = np.zeros((2,2,N))
         dLmda_dt[0,0,:] = -1/tau*(Lmda[0,0,:]-Lmda_0[0,0,:]) + R_0/(4*tau*mu*K)*((mu+K)*stress[0,0,:] + (mu-K)*stress[1,1,:])
         dLmda_dt[1,1,:] = -1/tau*(Lmda[1,1,:]-Lmda_0[1,1,:]) + R_0/(4*tau*mu*K)*((mu+K)*stress[1,1,:] + (mu-K)*stress[0,0,:])
         dLmda_dt[0,1,:] = -1/tau*(Lmda[0,1,:]-Lmda_0[0,1,:]) + R_0/(2*tau*mu)*stress[0,1,:]
         dLmda_dt[1,0,:] = -1/tau*(Lmda[1,0,:]-Lmda_0[1,0,:]) + R_0/(2*tau*mu)*stress[1,0,:]

         Lmda += dLmda_dt*dt

         lmda_major = 0.5*(Lmda[0,0,:]+Lmda[1,1,:]) + np.sqrt(0.25*(Lmda[0,0,:]-Lmda[1,1,:])**2 + Lmda[0,1,:]**2)
         lmda_minor = 0.5*(Lmda[0,0,:]+Lmda[1,1,:]) - np.sqrt(0.25*(Lmda[0,0,:]-Lmda[1,1,:])**2 + Lmda[0,1,:]**2)

         theta_def = 0.5*np.arctan2(2*Lmda[0,1,:],(Lmda[0,0,:]-Lmda[1,1,:]))
         theta_def = theta_def - np.pi*np.round((theta_def-theta)/(np.pi))   #periodic boundaries since theta_def is defined up to pi
         theta += torque_mech*dt + (theta_def-theta) + np.sqrt(2*D_r*dt)*np.random.normal(0.0, 1.0, N)
         theta = periodic_boundaries(theta, 2*np.pi)
    else :
         theta += torque_mech*dt + np.sqrt(2*D_r*dt)*np.random.normal(0.0, 1.0, N)
         theta = periodic_boundaries(theta, 2*np.pi)

    sin = np.sin(theta)
    cos = np.cos(theta)
    Lmda[0,0,:] = cos**2*lmda_major+sin**2*lmda_minor
    Lmda[1,1,:] = sin**2*lmda_major+cos**2*lmda_minor
    Lmda[0,1,:] = sin*cos*(lmda_major-lmda_minor)
    Lmda[1,0,:] = sin*cos*(lmda_major-lmda_minor)

    x += (v0*np.cos(theta) + Fx)*dt + np.sqrt(2*D*dt)*np.random.normal(0.0, 1.0, N)
    x = periodic_boundaries(x, L)
    y += (v0*np.sin(theta) + Fy)*dt + np.sqrt(2*D*dt)*np.random.normal(0.0, 1.0, N)
    y = periodic_boundaries(y, L)

    return x,y, theta, Lmda, lmda_major, lmda_minor


interval = 100      #used to be 250

traj_x = np.zeros((N,steps//interval))
traj_y = np.zeros((N,steps//interval))
traj_theta = np.zeros((N,steps//interval))
traj_lmda_major = np.zeros((N,steps//interval))
traj_lmda_minor = np.zeros((N,steps//interval))

def main(x,y, theta, Lmda, lmda_major, lmda_minor):
    for t in tqdm.tqdm(range(relax_steps)):
        x,y, theta, Lmda, lmda_major, lmda_minor = update(x,y, theta, Lmda, lmda_major, lmda_minor, lmda_major_0, lmda_minor_0, False, t)

    for t in tqdm.tqdm(range(steps)):
        x,y, theta, Lmda, lmda_major, lmda_minor = update(x,y, theta, Lmda, lmda_major, lmda_minor, lmda_major_0, lmda_minor_0, deformable, t)

        if t%interval==0:
            traj_x[:,t//interval] = x
            traj_y[:,t//interval] = y
            traj_theta[:,t//interval] = theta
            traj_lmda_major[:,t//interval] = lmda_major
            traj_lmda_minor[:,t//interval] = lmda_minor

print("Starting the simulation...")
main(x,y, theta, Lmda, lmda_major, lmda_minor)
print("Simulation ended..")
print("Plotting and creating movie (mp4)")

#initializing plot
#norm = plt.Normalize(-1, 0)    #for normalizing colormap
deform_ratio = traj_lmda_major/traj_lmda_minor
#norm = plt.Normalize(np.min(deform_ratio), np.max(deform_ratio))    #these are chosen somewhat arbitrarily for now
norm = plt.Normalize(1, 2)  #since lmda1>=lmda2 by definition, min=1. max is chosen by feel
cmap = LinearSegmentedColormap.from_list("CustomCmap", ["green", "yellow", "red"])
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(-L/2, L/2), ax.set_xticks([])
ax.set_ylim(-L/2, L/2), ax.set_yticks([])

def plot(t):
    #plotting
    ax.clear()
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    #ec = EllipseCollection(R_major*2, R_minor*2, theta/np.pi*180, units='x', offsets=np.array([x,y]).T,
                           #offset_transform=ax.transData, array=lmda_dif, cmap=cmap, norm=norm)
    ec = EllipseCollection(traj_lmda_major[:,t]*2, traj_lmda_minor[:,t]*2, traj_theta[:,t]/np.pi*180, units='x', offsets=np.array([traj_x[:,t],traj_y[:,t]]).T,
                           offset_transform=ax.transData, array=deform_ratio[:,t], cmap=cmap, norm=norm)
    ax.add_collection(ec)
    ax.quiver(traj_x[:,t], traj_y[:,t], np.cos(traj_theta[:,t]), np.sin(traj_theta[:,t]), color="black")
    return ax,


animation = FuncAnimation(fig, plot, frames=steps//interval, interval=500, blit=False, repeat=False)
#writervideo = FFMpegWriter(fps=20, codec="libx264", bitrate=1800,
#                      extra_args=["-pix_fmt", "yuv420p"],
#                      metadata={"artist": "Nina/Chinmay"})

writervideo = FFMpegWriter(
    fps=8,
    codec="libx264",
    extra_args=[
        "-crf", "22",          # 18–23 is common; lower = better quality
        "-preset", "fast",     # slower = better compression at same quality
        "-pix_fmt", "yuv420p"  # most compatible
    ],
    metadata={"artist": "Chinmay"},)
animation.save("movie.mp4", writer = writervideo, dpi=100)
#plt.show()

"""
param_template = Template(ellipse_template)

params = {
    "num_threads":5,
    "phi": 1.0,
    "dt": 0.001,
    "sim_time": 500,
    "boxl": 50,
    "act_vel": 0.2,
    "Diff_t": 0.0,
    "Diff_r": 0.01,
    "gb_epsilon": 1.0,
    "beta_exp": 2,
    "R_0": 1.0,
    "sigma_0": 2.0,
    "aspect_ratio": 1.2,
    "aspect_ratio_init": 1.2,
    "KV_tau": 1.0,
    "KV_mu": 0.5,
    "KV_K": 50.0,
    "deform_flag": True,
    "dt_relax": 0.001,
    "relax_steps": 10000,
    "seed": 12345
}

seeds = [22]
act_vels = [0.1, 0.2]
taus = [0.1,0.3,1,3,10,30,100]
mus = [0.1,0.3,1.0]

cur_dir = Path.cwd()
ellipse_code_path = f"/data1/pabshettiwar/Simulations/Ellipsoids/Automate_python_scheduler/Ellipsoids_harmonic.py"
software_path = "python3"
print(cur_dir)


for seed in seeds:
    for v0 in act_vels:
        for tau in taus:
            for mu in mus:
                    
                    if not os.path.isdir(os.path.join(cur_dir, f"Seed-{seed}")):
                        os.mkdir(os.path.join(cur_dir, f"Seed-{seed}"))
                        path = os.path.join(cur_dir, f"Seed-{seed}")
                    else:
                        path = os.path.join(cur_dir, f"Seed-{seed}")

                    if not os.path.isdir(os.path.join(path, f"phi_1.0_v0_{v0}_tau_{tau}_mu_{mu}") ):
                        os.mkdir(os.path.join(path, f"phi_1.0_v0_{v0}_tau_{tau}_mu_{mu}"))
                        path = os.path.join(path, f"phi_1.0_v0_{v0}_tau_{tau}_mu_{mu}")
                    else:
                        path = os.path.join(path,f"phi_1.0_v0_{v0}_tau_{tau}_mu_{mu}")

                    print("current_path=", path)

                    #shutil.copy2(input_file_path,path)
                    #print(f"copied to the above path")

                    with open(os.path.join(path,f"ellipse_harmonic.py"), "w") as config_file:
                        config_data = param_template.render(params)
                        config_file.write(config_data)

                    curr_file_path = os.path.join(path,f"ellipse_harmonic.py")

                    # check for the cpu usage to fall below threshold
                    wait_for_cpu_usage_below(threshold = 20, check_interval=300)

                    # Run the job in the background
                    run_in_background(software_path, curr_file_path, path)

                    time.sleep(100)


'''
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



