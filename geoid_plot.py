import numpy as np
from geoid_toolkit.read_ICGEM_harmonics import read_ICGEM_harmonics
from geoid_toolkit.topographic_potential import topographic_potential
from geoid_toolkit.real_potential import real_potential
from geoid_toolkit.norm_potential import norm_potential
from geoid_toolkit.norm_gravity import norm_gravity
from tqdm import tqdm  # Import tqdm for the progress bar
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from multiprocessing import Pool
import csv

# Define the boundaries
lon_min = -9.5  # 9º 30' W
lon_max = 4.5   # 4º 30' E
lat_min = 35.0  # 35º N
lat_max = 44.0  # 44º N

# Define the grid resolution (in degrees)
# 1 km is approximately equal to 0.009 degrees
lon_res = 0.009
lat_res = 0.009

# Generate the grid points for longitude and latitude
lon_grid = np.arange(lon_min, lon_max + lon_res, lon_res)
lat_grid = np.arange(lat_min, lat_max + lat_res, lat_res)

# Initialize an array to store the geoid undulations
geoid_undulations = np.zeros((len(lat_grid), len(lon_grid)))

# Parameters
lmax = 50
R = 6378136.3
GM = 3.986004415E+14

# Extracting static gravity model file coefficients
gravity_model_file = 'SGG-UGM-2.gfc'
Ylms = read_ICGEM_harmonics(gravity_model_file, TIDE='mean_tide', lmax=lmax, ELLIPSOID='WGS84')

clm = Ylms['clm']
slm = Ylms['slm']

def corrected_geoid_undulation(lat_lon):
    lat, lon = lat_lon
    refell = 'WGS84'
    tclm = model_input['clm']
    tslm = model_input['slm']
    density = model_input['density']
    GAUSS=0
    EPS=1e-8
    max_iterations=1000
    
    # calculate the real and normal potentials for the first iteration
    W, dWdr = real_potential(lat, lon, 0.0, refell, clm, slm, lmax, R, GM, GAUSS=GAUSS)
    U, dUdr, dUdt = norm_potential(lat, lon, 0.0, refell, lmax)
    # topographic potential correction
    T = topographic_potential(lat, lon, refell, tclm, tslm, lmax, R, density)
    # normal gravity at latitude
    gamma_h, dgamma_dh = norm_gravity(lat, 0.0, refell)
    # geoid height for first iteration
    N_1 = (W - U - T) / gamma_h
    # set geoid height to the first iteration and set RMS as infinite
    N = np.copy(N_1)
    RMS = np.inf

    iteration = 0
    while RMS > EPS and iteration < max_iterations:
        # calculate the real potentials for the iteration
        W, dWdr = real_potential(lat, lon, N_1, refell, clm, slm, lmax, R, GM, GAUSS=GAUSS)
        # add geoid height for iteration
        N_1 += (W - U - T) / gamma_h
        # calculate RMS between iterations
        RMS = np.sqrt(np.sum((N - N_1)**2) / len(lat))
        # set N to the previous iteration
        N = np.copy(N_1)
        iteration += 1

    if iteration == max_iterations:
        print("Warning: Maximum iterations reached before convergence.")

    return N[0]

def read_topography_harmonics(model_file):
    """
    Reads spherical harmonic coefficients from a .bshc file.

    Parameters
    ----------
    model_file: str
        Full path to the .bshc file with spherical harmonic coefficients.

    Returns
    -------
    model_input: dict
        Dictionary containing the spherical harmonic coefficients and model parameters.
    """
    dinput = np.fromfile(model_file, dtype=np.dtype('<f8'))
    
    header = 2
    input_lmin, input_lmax = dinput[:header].astype(np.int64)
    
    n_down = ((input_lmin - 1)**2 + 3*(input_lmin - 1)) // 2 + 1
    n_up = (input_lmax**2 + 3*input_lmax) // 2 + 1
    n_harm = n_up - n_down
    
    model_input = {}
    model_input['modelname'] = 'EARTH2014'
    model_input['density'] = 2670.0
    
    ii, jj = np.tril_indices(input_lmax + 1)
    
    model_input['l'] = np.arange(input_lmax + 1)
    model_input['m'] = np.arange(input_lmax + 1)
    model_input['clm'] = np.zeros((input_lmax + 1, input_lmax + 1))
    model_input['slm'] = np.zeros((input_lmax + 1, input_lmax + 1))
    model_input['clm'][ii, jj] = dinput[header:(header + n_harm)]
    model_input['slm'][ii, jj] = dinput[(header + n_harm):(header + 2 * n_harm)]
    
    return model_input

model_file = 'dV_ELL_EARTH2014_5480.bshc'
model_input = read_topography_harmonics(model_file)

tclm = model_input['clm']
tslm = model_input['slm']
density = model_input['density']

# Create a list of latitude and longitude pairs for parallel processing
lat_lon_pairs = [(lat_grid[i], lon_grid[j]) for i in range(len(lat_grid)) for j in range(len(lon_grid))]

# Use multiprocessing Pool to parallelize the calculation of geoid undulations
with Pool() as pool:
    results = list(tqdm(pool.imap(corrected_geoid_undulation, lat_lon_pairs), total=len(lat_lon_pairs), desc="Calculating geoid undulations"))

# Reshape the results back into the grid shape
geoid_undulations = np.array(results).reshape(len(lat_grid), len(lon_grid))

# Save the results to a CSV file
with open('geoid_undulations.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Longitude', 'Latitude', 'Geoid Undulation'])
    for i in range(len(lat_grid)):
        for j in range(len(lon_grid)):
            csvwriter.writerow([lon_grid[j], lat_grid[i], geoid_undulations[i, j]])

# Perform Kriging interpolation on the calculated geoid undulations

# Flatten the grid points and undulations for Kriging input
lon_flattened = np.repeat(lon_grid, len(lat_grid))
lat_flattened = np.tile(lat_grid, len(lon_grid))
undulations_flattened = geoid_undulations.flatten()

# Create Ordinary Kriging object and perform interpolation on a finer grid
OK = OrdinaryKriging(lon_flattened,
                     lat_flattened,
                     undulations_flattened,
                     variogram_model='linear',
                     verbose=False,
                     enable_plotting=False)

lon_fine = np.linspace(lon_min,
                       lon_max,
                       len(lon_grid) * 10)
lat_fine = np.linspace(lat_min,
                       lat_max,
                       len(lat_grid) * 10)

lon_fine_grid,
lat_fine_grid = np.meshgrid(lon_fine,
                            lat_fine)

z_fine,
ss_fine = OK.execute('grid',
                     lon_fine,
                     lat_fine)

# Plot the results
plt.figure(figsize=(10, 8))
plt.contourf(lon_fine_grid, lat_fine_grid, z_fine, cmap='viridis')
plt.colorbar(label='Geoid Undulation (m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geoid Undulations')
plt.show()
