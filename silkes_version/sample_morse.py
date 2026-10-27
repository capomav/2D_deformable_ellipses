

from ellipse_interface_v2 import *

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
