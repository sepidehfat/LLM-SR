import re
import json
from pathlib import Path

def extract_mini_equations(txt_path: str) -> list[str]:
    lines = Path(txt_path).read_text().splitlines()
    eqs, i = [], 0

    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        # Skip blank/header/footer/def/docstrings/comments
        if not s or "====" in s or "Evaluated Function" in s:
            i += 1
            continue
        if s.startswith(("def ", '"""', "'''", "#")):
            i += 1
            continue

        # Handle return on one line
        if s.startswith("return"):
            expr = s[len("return"):].strip().split("#", 1)[0].strip()
            if expr:
                eqs.append(expr)
            i += 1
            continue

        # Handle assignments/in-place updates, possibly multi-line
        if "=" in s:
            # start buffering this statement
            buffer = [s.split("#", 1)[0].rstrip()]
            open_parens = buffer[0].count("(") - buffer[0].count(")")
            j = i + 1
            while open_parens > 0 and j < len(lines):
                nxt = lines[j].strip()
                if nxt and not nxt.startswith("#"):
                    clean = nxt.split("#", 1)[0].rstrip()
                    buffer.append(clean)
                    open_parens += clean.count("(") - clean.count(")")
                j += 1
            full = " ".join(buffer).replace("  ", " ")
            eqs.append(full)
            i = j
            continue

        i += 1

    return eqs


def extract_first_docstring(txt_path: str) -> str:
    content = Path(txt_path).read_text()
    # capture first triple-quoted docstring
    m = re.search(r'"""\s*(.*?)\s*"""', content, flags=re.S)
    if not m:
        m = re.search(r"'''\s*(.*?)\s*'''", content, flags=re.S)
    return m.group(1).strip() if m else ""


def embed_equations_and_description(
    eq_txt: str,
    desc_txt: str,
    json_in: str,
    json_out: str = None
) -> list[dict]:
    # Extract all mini-equations including return expressions
    raw_eqs = extract_mini_equations(eq_txt)

    # Build numbered steps for all but the last
    steps = []
    for idx, expr in enumerate(raw_eqs[:-1], start=1):
        steps.append(f"Step {idx}: {expr}")

    # Determine the actual final equation
    if raw_eqs:
        final_candidate = raw_eqs[-1]
        # If it's just a variable name, fall back to previous expression
        if re.fullmatch(r"[A-Za-z_]\w*", final_candidate) and len(raw_eqs) > 1:
            actual_final = raw_eqs[-2]
        else:
            actual_final = final_candidate
        steps.append(f"Final Equation: {actual_final}")

    full_sequence = " ".join(steps)

    # Extract description
    docstring = extract_first_docstring(desc_txt)

    # Load JSON and embed
    records = json.loads(Path(json_in).read_text())
    for rec in records:
        rec["predicted_equations"] = full_sequence
        rec["description"] = docstring

    # Write back out
    target = Path(json_out) if json_out else Path(json_in)
    target.write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":

    model = "mixtral"
    module = "all"
    updated = embed_equations_and_description(
        eq_txt   = f"../Final_equation/{model}/{module}_best_function.txt",
        desc_txt = f"../specs/specification_{module}_numpy.txt",
        json_in  = f"./problems_file.json",
        json_out = f"./problems_with_metadata_{model}_{module}.json"
    )
    print("Done! Updated records:\n", updated)