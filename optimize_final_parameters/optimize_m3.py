import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAX_NPARAMS = 10
# @evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate the equation on data observations."""
    inputs, outputs = data['inputs'], data['outputs']
    
    # Unpack variables from inputs
    time, depth, ice, snow, snowice, temp_total05 = inputs.T

    # Loss function to optimize
    def loss(params):
        y_pred = equation(
            time, depth, ice, snow, snowice, temp_total05, params
        )
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')
    optimized_params = result.x
    loss_value = result.fun

    if np.isnan(loss_value) or np.isinf(loss_value):
        return None, optimized_params
    else:
        return -loss_value, optimized_params


# @equation.evolve
def equation(time: np.ndarray, depth: np.ndarray, ice: np.ndarray, snow: np.ndarray, snowice: np.ndarray, temp_total05: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate vertical diffusion impact on temperature using input profiles.
    """
    """
    Updated version of the vertical diffusion equation, considering the thermal resistance of the ice, snow, and snow-ice layers.
    """
    K = params[0]
    dz = np.gradient(depth, axis=0)
    
    # Calculate Q at each layer. Note that a factor (K / dz) is included in the expression
    diffusion_term = -K / (dz * dz) * (np.roll(temp_total05, -1) - 2 * temp_total05 + np.roll(temp_total05, 1))
    ice_term = -params[1] * ice / (params[4] * dz) if np.any(ice) else 0
    snow_term = -params[2] * snow / (params[5] * dz) if np.any(snow) else 0
    snowice_term = -params[3] * snowice / (params[6] * dz) if np.any(snowice) else 0
    
    # The sum of the terms Q
    Q = diffusion_term + ice_term + snow_term + snowice_term
    
    # Apply the discretized diffusion equation
    temp_total05 = temp_total05 + Q * np.gradient(dz)
    
    return temp_total05
    
# ✅ Step 3: Load dataset from train.csv
problem_name = 'm3'
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