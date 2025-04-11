import numpy as np
from geoid_toolkit.read_ICGEM_harmonics import read_ICGEM_harmonics
from geoid_toolkit.topographic_potential import topographic_potential
from geoid_toolkit.real_potential import real_potential
from geoid_toolkit.norm_potential import norm_potential
from geoid_toolkit.norm_gravity import norm_gravity
from tqdm import tqdm  # Import tqdm for the progress bar
import concurrent.futures  # Import concurrent.futures for parallelization

# Parameters
lon = np.array([-0.48123267, 0.0])  # Longitude array
lat = np.array([38.33891758, 0.0])  # Latitude array
lmax = 2190
R = 6378136.3
GM = 3.986004415E+14

# Extracting static gravity model file coefficients
gravity_model_file = 'SGG-UGM-2.gfc'
Ylms = read_ICGEM_harmonics(gravity_model_file, TIDE='mean_tide', lmax=lmax, ELLIPSOID='GRS80')

clm = Ylms['clm']
slm = Ylms['slm']


def corrected_geoid_undulation(lat, lon, refell, clm, slm, tclm, tslm, lmax,
                               R, GM, density, GAUSS=0, EPS=1e-8, max_iterations=1000):
    """
    Calculates the topographically corrected geoidal undulation at given
    latitudes and longitudes using an iterative approach.
    """

    def calculate_potentials(lat, lon, N_1):
        W, dWdr = real_potential(lat, lon, N_1, refell, clm, slm, lmax, R, GM, GAUSS=GAUSS)
        U, dUdr, dUdt = norm_potential(lat, lon, N_1, refell, lmax)
        return W, U

    # Calculate the topographic potential correction and normal gravity
    T = topographic_potential(lat, lon, refell, tclm, tslm, lmax, R, density)
    gamma_h, dgamma_dh = norm_gravity(lat, 0.0, refell)

    # Initialize geoid height for the first iteration
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_potentials, lat[i], lon[i], 0.0) for i in range(len(lat))]
        results = [future.result() for future in futures]

    W, U = zip(*results)
    W = np.array(W)
    U = np.array(U)
    N_1 = (W - U - T) / gamma_h
    N = np.copy(N_1)
    RMS = np.inf

    def iteration_step(lat, lon, N_1):
        W, dWdr = real_potential(lat, lon, N_1, refell, clm, slm, lmax, R, GM, GAUSS=GAUSS)
        N_1 += (W - U - T) / gamma_h
        return N_1

    with tqdm(desc="Calculating geoid undulation", unit="iteration") as pbar:
        iteration = 0
        while RMS > EPS and iteration < max_iterations:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(iteration_step, lat[i], lon[i], N_1[i]) for i in range(len(lat))]
                N_1 = np.array([future.result() for future in futures])

            RMS = np.sqrt(np.sum((N - N_1) ** 2) / len(lat))
            N = np.copy(N_1)
            iteration += 1
            pbar.update(1)

        pbar.total = iteration
        pbar.refresh()

        if iteration == max_iterations:
            print("Warning: Maximum iterations reached before convergence.")

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

    def read_file_chunk(start, end):
        return np.fromfile(model_file, dtype=np.dtype('<f8'), count=end - start, offset=start)

    header_size = 2 * np.dtype('<f8').itemsize
    dinput_header = np.fromfile(model_file, dtype=np.dtype('<f8'), count=2)
    input_lmin, input_lmax = dinput_header.astype(np.int64)
    n_down = ((input_lmin - 1) ** 2 + 3 * (input_lmin - 1)) // 2 + 1
    n_up = (input_lmax ** 2 + 3 * input_lmax) // 2 + 1
    n_harm = n_up - n_down

    model_input = {}
    model_input['modelname'] = 'EARTH2014'
    model_input['density'] = 2670.0

    ii, jj = np.tril_indices(input_lmax + 1)
    model_input['l'] = np.arange(input_lmax + 1)
    model_input['m'] = np.arange(input_lmax + 1)
    model_input['clm'] = np.zeros((input_lmax + 1, input_lmax + 1))
    model_input['slm'] = np.zeros((input_lmax + 1, input_lmax + 1))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_clm = executor.submit(read_file_chunk, header_size, header_size + n_harm * np.dtype('<f8').itemsize)
        future_slm = executor.submit(read_file_chunk, header_size + n_harm * np.dtype('<f8').itemsize,
                                     header_size + 2 * n_harm * np.dtype('<f8').itemsize)
        clm_data = future_clm.result()
        slm_data = future_slm.result()

    model_input['clm'][ii, jj] = clm_data
    model_input['slm'][ii, jj] = slm_data

    return model_input


# Example usage
model_file = 'dV_ELL_EARTH2014.bshc'
model_input = read_topography_harmonics(model_file)

tclm = model_input['clm']
tslm = model_input['slm']
density = model_input['density']

N = corrected_geoid_undulation(lat, lon, 'GRS80', clm, slm, tclm, tslm, lmax, R, GM, density)

print(N)
