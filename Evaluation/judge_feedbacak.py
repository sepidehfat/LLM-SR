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


def llm_judge(gt_equation: str, predicted_equation: str, X: List[List[float]]) -> str:
    """
    Query the LLM to judge whether the predicted equation is correct.
    """
    prompt = f"""
            You are an expert in symbolic mathematics and physical modeling.

            **Task**: Given the mathematical function of the process, review the logic of the given function to see if it aligns with the physical description below:

            Analyse the structure and form of expressions analytically.

            Please provide a step-by-step explanation of your reasoning, followed by your final answer.

            Format:
            Reasoning: <your explanation here>
            Answer: <Yes/No>

            Expression A (Ground Truth):
            {gt_equation}

            Expression B (Predicted):
            {predicted_equation}
            """


    result = query_llama3_local(prompt)

    print("----- Raw LLM Output -----")
    print(result)
    print("--------------------------")

    # Extract answer using regex for robustness
    match = re.search(r'Answer\s*:\s*(Yes|No)', result, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unclear"

def evaluate_batch(problems_file: str, output_file: str):
    with open(problems_file, 'r') as f:
        problems = json.load(f)

    results = []

    for problem in problems:
        eq_id = problem['equation_id']
        gt = problem['gt_equation']
        preds = problem['predicted_equations']
        X = problem.get('X_samples', [])

        print(f"Evaluating Problem {eq_id}: GT = {gt}")

        judgments = []
        for pred in preds:
            judgment = llm_judge(gt, pred, X)
            print(f"  ↳ Prediction: {pred} → {judgment}")
            judgments.append({
                "predicted_equation": pred,
                "judgment": judgment
            })

        results.append({
            "equation_id": eq_id,
            "gt_equation": gt,
            "judgments": judgments
        })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

evaluate_batch("./problems_file.json","./eval_results.json")