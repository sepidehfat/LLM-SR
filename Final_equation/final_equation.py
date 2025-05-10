import re


data = "all"
# Define the file path 
file_path = f"../logs/llama1/{data}/{data}.txt"  
output_file = f"./llama/{data}_best_function.txt"


# Regular expressions to extract scores and function definitions
score_pattern = re.compile(r'Score\s*:\s*([-+]?\d*\.\d+e?[-+]?\d*|None)')
function_pattern = re.compile(r'================= Evaluated Function =================')

# Initialize tracking variables
closest_to_zero_score = float('-inf')  # Start with the smallest possible value
best_function = None
current_function = []
capture_function = False

# Read and process the file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Detect function start
        if function_pattern.search(line):
            capture_function = True
            current_function = []  # Start capturing a new function

        # Capture function definition
        if capture_function:
            current_function.append(line)

        # Detect function end (marked by "------------------------------------------------------")
        if line.strip() == "------------------------------------------------------":
            capture_function = False  # Stop capturing

        # Extract scores
        match = score_pattern.search(line)
        if match:
            score_value = match.group(1).strip()
            
            # Convert score to float, ignoring 'None'
            if score_value != "None":
                score = float(score_value)
                
                # We want the smallest negative number (the largest value among negatives)
                if score < 0 and score > closest_to_zero_score:
                    closest_to_zero_score = score
                    best_function = "".join(current_function)  # Store function code

# Output the function with the smallest negative score (closest to zero)
if best_function:
    print(f"Function with the score closest to zero ({closest_to_zero_score}):\n")
    print(best_function)
    
    with open(output_file, "w") as f:
        f.write(best_function)
else:
    print("No valid negative scores found in the file.")

