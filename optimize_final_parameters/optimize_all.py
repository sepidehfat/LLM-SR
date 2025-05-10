import numpy as np
import pandas as pd
from scipy.optimize import minimize

# @evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate full end-to-end lake model over one timestep."""
    inputs, outputs = data['inputs'], data['outputs']

    (time, depth, AirTemp_degC, Longwave_Wm2, Latent_Wm2, Sensible_Wm2,
    Shortwave_Wm2, lightExtinct_m1, Area_m2, ice, snow, snowice, buoyancy,
    diffusivity, temp_initial00) = inputs.T

    def loss(params):
        y_pred = equation(
        time, depth, AirTemp_degC, Longwave_Wm2, Latent_Wm2, Sensible_Wm2,
        Shortwave_Wm2, lightExtinct_m1, Area_m2, ice, snow, snowice, buoyancy,
        diffusivity, temp_initial00, params
        )
        return np.mean((y_pred - outputs) ** 2)
    
    MAX_NPARAMS = 15
    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None, optimized_params
    else:
        return -loss, optimized_params


# @equation.evolve
def equation(time: np.ndarray, depth: np.ndarray, AirTemp_degC: np.ndarray, Longwave_Wm2: np.ndarray, Latent_Wm2: np.ndarray, Sensible_Wm2: np.ndarray, Shortwave_Wm2: np.ndarray, lightExtinct_m1: np.ndarray, Area_m2: np.ndarray, ice: np.ndarray, snow: np.ndarray, snowice: np.ndarray, buoyancy: np.ndarray, diffusivity: np.ndarray, temp_initial00: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict final lake temperature for the next <time, depth> pair after full process chain using symbolic form.
    """
    
    # Calculate radiation heat flux
    radiation_heat_flux = params[0] * (Shortwave_Wm2 + Longwave_Wm2)
    
    # Calculate convection term
    convection_term = params[1] * (AirTemp_degC - temp_initial00)
    
    # Calculate vertical heat diffusion
    diffusivity_term = params[2] * diffusivity * (temp_initial00 - AirTemp_degC)
    
    # Calculate lake depth heating term
    lake_depth_heating_term = params[3] * depth * (AirTemp_degC - temp_initial00)
    
    # Calculate lake surface heating term
    lake_surface_heating_term = params[4] * (Shortwave_Wm2 - Latent_Wm2 - Sensible_Wm2)
    
    # Calculate ice melting term
    ice_melting_term = params[5] * (snow + ice) * (AirTemp_degC - params[6])
    
    # Calculate thermal insulation term
    thermal_insulation_term = params[7] * (snow + ice) * (AirTemp_degC - temp_initial00)
    
    # Calculate buoyancy term
    buoyancy_term = params[8] * buoyancy * (temp_initial00 - AirTemp_degC)
    
    # Calculate other heat fluxes (e.g. latent, sensible)
    other_heat_fluxes_term = params[9] * (Latent_Wm2 + Sensible_Wm2)
    
    # Calculate final lake temperature
    temp_update = (1 - params[10]) * temp_initial00 + radiation_heat_flux + convection_term + diffusivity_term + lake_depth_heating_term + lake_surface_heating_term + ice_melting_term + thermal_insulation_term + buoyancy_term + other_heat_fluxes_term
    
    return temp_update



# ✅ Step 3: Load dataset from train.csv
problem_name = 'all'
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