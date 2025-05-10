import numpy as np
import pandas as pd
from scipy.optimize import minimize

# @evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    
    (time, depth, AirTemp_degC, Longwave_Wm, Latent_Wm, Sensible_Wm,
    Shortwave_Wm, lightExtinct_m, Area_m2, ice, snow, snowice, temp_initial00) = inputs.T
    
    # Optimize parameters based on data
    def loss(params):
        y_pred = equation(
        time, depth, AirTemp_degC, Longwave_Wm, Latent_Wm, Sensible_Wm,
        Shortwave_Wm, lightExtinct_m, Area_m2, ice, snow, snowice, temp_initial00,
        params
         )
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    MAX_NPARAMS = 10
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None, optimized_params
    else:
        return -loss, optimized_params


# @equation.evolve
def equation(time: np.ndarray, depth: np.ndarray, AirTemp_degC: np.ndarray, Longwave_Wm: np.ndarray, Latent_Wm: np.ndarray, Sensible_Wm: np.ndarray, Shortwave_Wm: np.ndarray, lightExtinct_m: np.ndarray, Area_m2: np.ndarray, ice: np.ndarray, snow: np.ndarray, snowice: np.ndarray, temp_initial00: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric approximation of lake temperature evolution based on energy balance.

    Returns:
        Updated temperature array (1D).
    """
    """
    Parametric approximation of lake temperature evolution based on energy balance.
    
    Returns:
        Updated temperature array (1D).
    """
    # Importance of atmospheric forcing
    dT_dt = params[0] * (AirTemp_degC - temp_initial00)
    
    # Exponential depth dependence
    temperature_profile = temp_initial00 + params[1] * depth**params[2]
    
    # Radiation absorption and scattering
    radiation_gradient = params[3] * (Longwave_Wm + Shortwave_Wm) * lightExtinct_m
    
    # Surface heat fluxes
    latent_heat_flux = Latent_Wm * 1000
    sensible_heat_flux = Sensible_Wm * 1000
    
    # Total heat flux and its impact on temperature profile
    total_heat_flux = radiation_gradient - (latent_heat_flux + sensible_heat_flux)
    temperature_profile += params[4] * total_heat_flux / (Area_m2 * params[5] * (1 + params[6] * (depth**2)))
    
    # Influence of ice, snow, and their combined effect
    icelimit = 0.0
    if ice.sum() > 0:
        icelimit = 0.0
    else:
        icelimit = -10.0
    
    if snow.sum() > 0:
        temperature_profile = np.where(depth > 0, temperature_profile, temp_initial00)
    else:
        temperature_profile -= params[7] * (snowice - icelimit)
    
    # Interplay between temperature and its derivatives
    d2T_dt2 = params[8] * (temperature_profile - temperature_profile)
    
    # Final temperature update
    temperature_profile += params[9] * (dT_dt + d2T_dt2)
    
    # Return the updated temperature profile
    return temperature_profile

# ✅ Step 3: Load dataset from train.csv
problem_name = 'm1'
df = pd.read_csv(f'./data/{problem_name}/train.csv')
data = np.array(df)

# Split into features (X) and labels (y)
X, y = data[:, :-1], data[:, -1].reshape(-1)
data_dict = {'inputs': X, 'outputs': y}

# ✅ Step 4: Run the optimization function and print parameters
score, optimized_params = evaluate(data_dict)

# ✅ Step 5: Print final results
print("\n✅ Evaluation Score (Negative MSE):", score)
print("✅ Optimized Parameters:", optimized_params)


def test_model(test_file, label):
    """Evaluate equation on a test dataset and compute MSE."""
    try:
        df_test = pd.read_csv(f'./data/{problem_name}/{test_file}')
        data_test = np.array(df_test)
        X_test, y_test = data_test[:, :-1], data_test[:, -1].reshape(-1)

        predictions = equation(*X_test.T, optimized_params)

        mse = np.mean((predictions - y_test) ** 2)

        # Check for invalid values in predictions
        if np.isnan(mse) or np.isinf(mse):
            print(f"⚠️ WARNING: {label} MSE contains invalid values! Skipping...")
        else:
            print(f"✅ {label} MSE: {mse}")

    except FileNotFoundError:
        print(f"⚠️ WARNING: {test_file} not found! Skipping {label} test.")

# ✅ Step 5: Run Tests on ID and OOD Data
test_model('test_id.csv', "Test_ID")
test_model('test_ood.csv', "Test_OOD")