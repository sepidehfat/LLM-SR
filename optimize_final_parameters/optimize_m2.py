import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAX_NPARAMS = 10
# @evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate the equation on data observations."""
    inputs, outputs = data['inputs'], data['outputs']

    (time, depth, Area_m2, Uw,
     buoyancy, ice, snow, snowice,
     diffusivity, temp_heat01) = inputs.T

    # Optimize parameters based on data
    def loss(params):
        y_pred = equation(
            time, depth, Area_m2, Uw,
            buoyancy, ice, snow, snowice,
            diffusivity, temp_heat01, params  # ← fixed missing comma
        )
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None, optimized_params
    else:
        return -loss, optimized_params


# @equation.evolve
def equation(time: np.ndarray, depth: np.ndarray, Area_m2: np.ndarray, Uw: np.ndarray, buoyancy: np.ndarray, ice: np.ndarray, snow: np.ndarray, snowice: np.ndarray, diffusivity: np.ndarray, temp_heat01: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Return updated surface temperature after accounting for ice/snow effects.
    """
    """
    Update lake surface temperature after accounting for ice/snow effects.
    """
    
    # Calculate changes in ice thickness
    ice_growth_rate = params[0] * (temp_heat01 - buoyancy) * (Area_m2 / (Area_m2 + ice + snow))
    new_ice = np.clip(ice + params[1] * ice_growth_rate * (Area_m2 / (Area_m2 + ice + snow)), 0, None)
    
    # Calculate changes in snowpack thickness
    snow_growth_rate = params[2] * (temp_heat01 - params[3]) * (Area_m2 / (Area_m2 + ice + snow))
    new_snow = np.clip(snow + params[4] * snow_growth_rate * (Area_m2 / (Area_m2 + ice + snow)), 0, None)
    
    # Calculate conduction heat loss from the lake surface
    conduction_loss = params[5] * diffusivity * (temp_heat01 - params[6])
    
    # Calculate lake surface temperature updates
    dT_surface = params[7] * (new_ice - ice) + params[8] * (new_snow - snow) - conduction_loss
    
    # Add heating/cooling effects of growing/melting ice and snow
    if snowice.any():  # Checking if snowice exists
        dT_surface += params[9] * params[7] * (new_ice - ice)  # Add heat loss from growing ice
    elif np.any(new_ice < ice):
        dT_surface -= params[9] * params[7] * (new_ice - ice)  # Add heat gain from melting ice
    else:
        dT_surface -= params[9] * params[7] * (new_ice - ice) + params[9] * params[7] * (new_snow - snow)  # Both ice and snow
    
    # Update lake surface temperature
    new_surface_temp = temp_heat01 + dT_surface
    
    return new_surface_temp
# ✅ Step 3: Load dataset from train.csv
problem_name = 'm2'
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