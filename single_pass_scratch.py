import numpy as np
import torch
import os
import time

from llmsr import config as config_lib
from llmsr.sampler import LocalLLM, _extract_body
from llmsr.evaluator import Evaluator, LocalSandbox
from llmsr import code_manipulation
import pandas as pd 
from scipy.optimize import minimize



os.environ["LLMSR_PORT"] = "5050"

# ------------------------------------------------------------------------------
# ✅ Helper to evaluate a sample and return its score
# ------------------------------------------------------------------------------
def evaluate_sample_return_score(evaluator, sample: str) -> float | None:
    from llmsr.evaluator import _sample_to_program, _calls_ancestor

    new_function, program = _sample_to_program(
        sample,
        version_generated=None,
        template=evaluator._template,
        function_to_evolve=evaluator._function_to_evolve
    )

    test_output, runs_ok = evaluator._sandbox.run(
        program=program,
        function_to_run=evaluator._function_to_run,
        function_to_evolve=evaluator._function_to_evolve,
        inputs=evaluator._inputs,
        test_input="train",
        timeout_seconds=evaluator._timeout_seconds
    )
    # breakpoint()
    if runs_ok and not _calls_ancestor(program, evaluator._function_to_evolve) and test_output is not None:
        if isinstance(test_output, (float, int)):
            return test_output
        else:
            raise ValueError("evaluate(...) did not return a numeric score.")
    return None

# ------------------------------------------------------------------------------
# Setup data
# ------------------------------------------------------------------------------

problem_name = 'all'
df = pd.read_csv(f'./data/{problem_name}/train.csv')
data = np.array(df)

X, y = data[:, :-1], data[:, -1].reshape(-1)
data_dict = {'inputs': X, 'outputs': y}

# ------------------------------------------------------------------------------
# Config for local LLM
# ------------------------------------------------------------------------------
my_config = config_lib.Config(
    use_api=False,
    api_model="gpt-3.5-turbo",
    num_evaluators=1,
    num_samplers=1,
    samples_per_prompt=1,
    evaluate_timeout_seconds=30,
)

# ------------------------------------------------------------------------------
# Load the full specification
# ------------------------------------------------------------------------------
with open(os.path.join(f"./specs/specification_{problem_name}_numpy.txt"), encoding="utf-8") as f:
    specification = f.read()

program = code_manipulation.text_to_program(specification)
initial_equation_body = program.get_function("equation").body

# ------------------------------------------------------------------------------
# Track equation versions manually
# ------------------------------------------------------------------------------
buffer = [initial_equation_body]

# ------------------------------------------------------------------------------
# Evaluator setup
# ------------------------------------------------------------------------------
evaluator = Evaluator(
    database=None,
    template=program,
    function_to_evolve="equation",
    function_to_run="evaluate",
    inputs={"train": data_dict},
    timeout_seconds=my_config.evaluate_timeout_seconds,
    sandbox_class=LocalSandbox,
)

# ------------------------------------------------------------------------------
# ✅ Evaluate initial equation
# ------------------------------------------------------------------------------
score_initial = evaluate_sample_return_score(evaluator, initial_equation_body)
print("✅ Initial equation score:", score_initial)
# ------------------------------------------------------------------------------
# Construct prompt from latest equation version
# ------------------------------------------------------------------------------
last_body = buffer[-1]
program_cp = code_manipulation.text_to_program(specification)
program_cp.get_function("equation").body = last_body
prompt_code = str(program_cp)

# ------------------------------------------------------------------------------
# Sample new candidate from LLM
# ------------------------------------------------------------------------------
llm = LocalLLM(samples_per_prompt=my_config.samples_per_prompt)
llm._instruction_prompt = "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n"

print("🌀 Sampling from LLM...\n")
t0 = time.time()
completions = llm.draw_samples(prompt_code, my_config)
dt = time.time() - t0
print(f"✅ Got {len(completions)} sample(s) in {dt:.2f} seconds.\n")

# ------------------------------------------------------------------------------
# ✅ Evaluate the sampled equation
# ------------------------------------------------------------------------------
sample_code = completions[0]
score = evaluate_sample_return_score(evaluator, sample_code)
print("✅ Sample score:", score)

# ------------------------------------------------------------------------------
# Save clean equation
# ------------------------------------------------------------------------------
clean_code = _extract_body(sample_code, my_config)
# with open(f"equation_candidate_{problem_name}.txt", "w", encoding="utf-8") as f:
#     f.write(sample_code)

# Track for next round (if needed)
buffer.append(sample_code)

# Print final code
print("\n📌 Final cleaned code:\n", sample_code)

# breakpoint()
