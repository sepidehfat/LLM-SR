import json
import time
from typing import List, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re


# Load model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_pipeline = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

def query_llama3_local(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are an expert in symbolic mathematics."},
        {"role": "user", "content": prompt.strip()}
    ]

    terminators = [
        llm_pipeline.tokenizer.eos_token_id,
        llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    output = llm_pipeline(
        messages,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
        return_full_text=False
    )

    # Extract only the assistant's content
    generated = output[0]["generated_text"]
    if isinstance(generated, list):  # In case you get a list of message dicts
        for item in generated:
            if isinstance(item, dict) and item.get("role") == "assistant":
                return item.get("content", "")
    elif isinstance(generated, str):
        return generated

    # Fallback
    return str(generated)


def llm_judge(predicted_equation: str, description: str) -> str:
    
    """
    Query the LLM to judge whether the predicted equation is 
    physically consistent with the process described.
    Returns 'yes', 'no', or 'unclear'.
    """
    parts     = predicted_equation.split("Final Equation:")
    steps_part = parts[0].strip()
    final_eq   = parts[1].strip() if len(parts) > 1 else steps_part
    prompt = f"""
            You are a scientific modelling expert.

            **Process description**  
            {description}

            **Model derivation steps**  
            {steps_part}

            **Final equation to evaluate**  
            {final_eq}

            **Task**  
            Evaluate whether the **Final equation** above is *physically consistent* with the description and derivation.  
            - Check dimensional consistency of every term.  
            - Verify correct dependence on each variable.  
            - Ensure sign conventions and conservation principles make sense.
            - Rate the overall performance based on how much it is a good represantative of the problem.

            Please show your step-by-step reasoning, then answer exactly:

            Reasoning: <your explanation here>  
            Physically Consistent: <Yes/No>
            Rating: <0-10>
            """
      # append the equation to evaluate
#     print(prompt)
#     print(predicted_equation)

    result = query_llama3_local(prompt)

    print("----- Raw LLM Output -----")
    print(result)
    print("--------------------------")

   # extract the final yes/no
    m = re.search(r'Physically Consistent\s*:\s*(Yes|No)', result, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return "unclear"

def evaluate_batch(problems_file: str, output_file: str):
    
    problems_file = f"./problems_with_metadata_{model}_{module}.json"
    output_file = f"./problems_with_metadata_{model}_{module}.txt"
    with open(problems_file, 'r') as f:
        problems = json.load(f)

    results = []

    for problem in problems:
        eq_id = problem['equation_id']
        preds = problem['predicted_equations']
        X = problem.get('X_samples', [])
        descp = problem.get('description')
#         print(preds)

        judgments = []
        
        judgment = llm_judge(preds, descp)
        print(f"  ↳ Prediction: {preds} → {judgment}")
        judgments.append({
            "predicted_equation": preds,
            "judgment": judgment
        })

    results.append({
        "equation_id": eq_id,
        "judgments": judgments
    })

    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2)

    print("----- Results -----")
    print(results)
    print("--------------------------")


model= "llama"
module = "all"
evaluate_batch(model, module)
