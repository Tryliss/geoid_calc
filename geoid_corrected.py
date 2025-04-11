import numpy as np
from geoid_toolkit.read_ICGEM_harmonics import read_ICGEM_harmonics
from geoid_toolkit.topographic_potential import topographic_potential
from geoid_toolkit.real_potential import real_potential
from geoid_toolkit.norm_potential import norm_potential
from geoid_toolkit.norm_gravity import norm_gravity
from tqdm import tqdm  # Import tqdm for the progress bar



#parameters
lon = np.array([-0.48123267])  # Longitude
lat = np.array([38.33891758])  # Latitude
lmax=2190
R=6378136.3
GM=3.986004415E+14


#Extracting static gravity model file coefficients
gravity_model_file='/home/christian/SGG-UGM-2.gfc'
Ylms = read_ICGEM_harmonics(gravity_model_file,TIDE='mean_tide',lmax=lmax,ELLIPSOID='GRS80')

#print(Ylms)

clm = Ylms['clm']
slm = Ylms['slm']
#print(clm)
#print(slm)

def corrected_geoid_undulation(lat, lon, refell, clm, slm, tclm, tslm, lmax,
    R, GM, density, GAUSS=0, EPS=1e-8, max_iterations=1000):
    """
    Calculates the topographically corrected geoidal undulation at a given
    latitude and longitude using an iterative approach.
    """
    # calculate the real and normal potentials for the first iteration
    print("Calculating real potential for the first iteration...")
    W, dWdr = real_potential(lat, lon, 0.0, refell, clm, slm, lmax, R, GM, GAUSS=GAUSS)
    print("Calculating normal potential for the first iteration...")
    U, dUdr, dUdt = norm_potential(lat, lon, 0.0, refell, lmax)
    # topographic potential correction
    print("Calculating topographic potential correction...")
    T = topographic_potential(lat, lon, refell, tclm, tslm, lmax, R, density)
    # normal gravity at latitude
    print("Calculating normal gravity at latitude...")
    gamma_h, dgamma_dh = norm_gravity(lat, 0.0, refell)
    # geoid height for first iteration
    N_1 = (W - U - T) / gamma_h
    # set geoid height to the first iteration and set RMS as infinite
    N = np.copy(N_1)
    RMS = np.inf

    # Initialize the progress bar with dynamic total iterations
    with tqdm(desc="Calculating geoid undulation", unit="iteration") as pbar:
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
            # Update the progress bar
            iteration += 1
            pbar.update(1)

        # Set the total iterations in the progress bar
        pbar.total = iteration
        pbar.refresh()

        # Check if the loop exited due to reaching max_iterations
        if iteration == max_iterations:
            print("Warning: Maximum iterations reached before convergence.")

    # return the geoid height
    return N

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
    # Read binary data from the .bshc file
    dinput = np.fromfile(model_file, dtype=np.dtype('<f8'))
    
    # Extract minimum and maximum spherical harmonic degree
    header = 2
    input_lmin, input_lmax = dinput[:header].astype(np.int64)
    
    # Number of spherical harmonic records for Clm and Slm
    n_down = ((input_lmin - 1)**2 + 3*(input_lmin - 1)) // 2 + 1
    n_up = (input_lmax**2 + 3*input_lmax) // 2 + 1
    n_harm = n_up - n_down
    
    # Dictionary of model parameters and output Ylms
    model_input = {}
    model_input['modelname'] = 'EARTH2014'
    model_input['density'] = 2670.0
    
    # Extract cosine and sine harmonics
    ii, jj = np.tril_indices(input_lmax + 1)
    
    # Output dimensions
    model_input['l'] = np.arange(input_lmax + 1)
    model_input['m'] = np.arange(input_lmax + 1)
    model_input['clm'] = np.zeros((input_lmax + 1, input_lmax + 1))
    model_input['slm'] = np.zeros((input_lmax + 1, input_lmax + 1))
    model_input['clm'][ii, jj] = dinput[header:(header + n_harm)]
    model_input['slm'][ii, jj] = dinput[(header + n_harm):(header + 2 * n_harm)]
    
    return model_input

# Example usage
model_file = '/home/christian/dV_ELL_EARTH2014.bshc'
model_input = read_topography_harmonics(model_file)

# Extracting the cosine and sine harmonics
tclm = model_input['clm']
tslm = model_input['slm']
density = model_input['density']

# Displaying the results
#print(f"Model Name: {model_input['modelname']}")
#print(f"Density: {density}")
#print("Cosine Harmonics:")
#print(tclm)
#print("Sine Harmonics:")
#print(tslm)



#computing the corrected geoid undulation
tclm = model_input['clm']
tslm = model_input['slm']
#print(lat)
#print(lon)

N = corrected_geoid_undulation(lat, lon, 'GRS80', clm, slm, tclm, tslm, lmax, R, GM, density)

print(N)
