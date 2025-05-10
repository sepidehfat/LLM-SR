import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAX_NPARAMS = 10
# @evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    
    time, depth, ice, snow, snowice, temp_diff02 = inputs.T
    
    # Optimize parameters based on data
    def loss(params):
        y_pred = equation(
            time, depth, ice, snow, snowice, temp_diff02, params
        )
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    MAX_NPARAMS = 10
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None,optimized_params
    else:
        return -loss,optimized_params


# @equation.evolve
def equation(time: np.ndarray, depth: np.ndarray, ice: np.ndarray, snow: np.ndarray, snowice: np.ndarray, temp_diff02: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Update lake temperature based on vertical convective mixing.
    """
    """
    Update lake temperature based on vertical convective mixing.
    alpha_fresh, alpha_salt, delta_fresh, delta_salt, alpha_density, beta_fresh, beta_salt, sigma_fresh, sigma_salt, c_coriolis = params
    """
    alpha_fresh, alpha_salt, delta_fresh, delta_salt, alpha_density, beta_fresh, beta_salt, sigma_fresh, sigma_salt, c_coriolis = params
    
    dTdz_fp = alpha_fresh * (temp_diff02 - np.roll(temp_diff02, -1, axis=0)) / depth
    dTdz_sp = alpha_salt * (temp_diff02 - np.roll(temp_diff02, 1, axis=0)) / depth
    
    return alpha_density * temp_diff02 + sigma_fresh * (beta_fresh * dTdz_fp / depth) * (1 - (beta_fresh * (ice / depth) * snowice)) + sigma_salt * (beta_salt * dTdz_sp / depth) * (1 - (beta_salt * (snow / depth) * snowice)) - delta_salt * beta_salt * dTdz_sp + c_coriolis * (beta_fresh * (ice / depth) * snowice)

    
# ✅ Step 3: Load dataset from train.csv
problem_name = 'm4'
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