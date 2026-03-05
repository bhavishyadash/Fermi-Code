import os
import re
import math
import pandas as pd
from typing import Optional, Tuple, List, Dict

# ----------------------------
# FP score definition
# ----------------------------
def fp_score(A: float, Aprime: float) -> float:
    """
    fp_score = max(0, 1 - (1/3) * |log10(A'/A)|)
    """
    if A <= 0 or Aprime <= 0:
        return float("nan")
    s = 1.0 - (1.0 / 3.0) * abs(math.log10(Aprime / A))
    return max(0.0, s)

def fp_avg_for_range(A: float, low: float, high: float) -> float:
    return (fp_score(A, low) + fp_score(A, high)) / 2.0


# ----------------------------
# Parsing helpers
# ----------------------------
def _clean_text(x: str) -> str:
    s = str(x).strip()
    # normalize different dash characters to '-'
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # remove commas and currency symbol
    s = s.replace(",", "").replace("$", "")
    # normalize words
    s = s.lower()
    s = s.replace("percentage", "percent")
    return s

def parse_number(text: str) -> Optional[float]:
    """
    Parses numbers from strings like:
    '30%', '550,000', '$35,000', '69 billion', '42 million', '645M', '5.5B / day', '28.57 percent'
    Returns float in base units. Percent stays as "percent points" (e.g., 30% -> 30.0)
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None

    s = _clean_text(text)
    if not s:
        return None

    # drop trailing "/ day" or similar
    s = re.sub(r"\s*/\s*day.*$", "", s)

    # percent forms
    s = s.replace(" percent", "%").replace("percent", "%")

    # word multipliers
    word_mult = 1.0
    if "trillion" in s:
        word_mult *= 1e12
        s = s.replace("trillion", "")
    if "billion" in s:
        word_mult *= 1e9
        s = s.replace("billion", "")
    if "million" in s:
        word_mult *= 1e6
        s = s.replace("million", "")
    if "thousand" in s:
        word_mult *= 1e3
        s = s.replace("thousand", "")

    # suffix multipliers (K/M/B/T) ONLY if immediately attached or separated by space
    # e.g. "645m", "5.5 b", "3.7t"
    m = re.search(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*([kmbt])\b", s)
    if m:
        val = float(m.group(1))
        suf = m.group(2)
        suf_mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[suf]
        return val * suf_mult  # suffix typically implies the full scale already

    # percent (return percent points, like 30 for 30%)
    m = re.search(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*%", s)
    if m:
        return float(m.group(1))

    # plain number
    m = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s)
    if not m:
        return None

    return float(m.group(0)) * word_mult

def parse_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses low/high from strings like:
    '8.44–9.74 percent', '4.25-5.375', '630M', '785M-15.7B', '28.57–28.57 percent', '55–75 billion'
    If only one value, returns (v, v).
    Also propagates a unit on the RIGHT side to the LEFT side when needed (e.g., '55–75 billion').
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return (None, None)

    raw = str(text).strip()
    if not raw:
        return (None, None)

    s = _clean_text(raw)

    if "-" not in s:
        v = parse_number(s)
        return (v, v)

    left, right = s.split("-", 1)
    left = left.strip()
    right = right.strip()

    # If the right side contains a word unit and left doesn't, copy it over.
    for unit_word in ["trillion", "billion", "million", "thousand", "%", "usd", "count"]:
        if unit_word in right and unit_word not in left:
            left = f"{left} {unit_word}"

    # If right side uses suffix (K/M/B/T) and left doesn't, copy that suffix
    # Only copy if right actually has a number+suffix pattern (prevents the 'percent'->'t' bug).
    suf_match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*([kmbt])\b", right)
    if suf_match and not re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*[kmbt]\b", left):
        left = f"{left}{suf_match.group(1)}"

    low = parse_number(left)
    high = parse_number(right)

    if low is None and high is not None:
        low = high
    if high is None and low is not None:
        high = low

    return (low, high)


# ----------------------------
# Column pairing logic
# ----------------------------
def infer_llm_name_from_path(path: str) -> str:
    # "DeepSeek-Table 1.csv" -> "DeepSeek"
    return os.path.basename(path).split("-")[0].strip()

def pair_range_and_fp_columns(columns: List[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (range_col, fp_col) pairs.
    This is tailored to your CSVs:
      - range cols contain 'pred_range' or 'Ans w/o'
      - fp cols contain 'fp'
      - match 'with image' vs 'w/o'
    """
    range_cols = [c for c in columns if ("pred_range" in c.lower()) or ("ans w/o" in c.lower())]
    fp_cols = [c for c in columns if "fp" in c.lower()]

    pairs = []
    for fp_col in fp_cols:
        fp_l = fp_col.lower()
        want_with = "with image" in fp_l
        want_wo = ("w/o" in fp_l) or ("without" in fp_l)

        best = None
        best_score = -10

        for rcol in range_cols:
            r_l = rcol.lower()
            score = 0

            # match mode
            if want_with and "with image" in r_l:
                score += 5
            if want_wo and ("ans w/o" in r_l or "w/o" in r_l or "without" in r_l):
                score += 5

            # penalize mismatched mode
            if want_with and ("ans w/o" in r_l or "w/o" in r_l):
                score -= 5
            if want_wo and "with image" in r_l:
                score -= 5

            # minor preference
            if "pred_range" in r_l:
                score += 1
            if "ans w/o" in r_l:
                score += 1

            if score > best_score:
                best_score = score
                best = rcol

        if best is not None and best_score > 0:
            pairs.append((best, fp_col))

    # dedupe by fp_col
    seen = set()
    out = []
    for r, f in pairs:
        if f not in seen:
            seen.add(f)
            out.append((r, f))
    return out


# ----------------------------
# Main checker
# ----------------------------
def check_fp_scores(csv_paths: List[str], tolerance: float = 1e-4) -> pd.DataFrame:
    """
    Returns a dataframe of rows where fp_given != fp_expected (outside tolerance).
    """
    bad_rows = []

    for path in csv_paths:
        df = pd.read_csv(path)
        llm = infer_llm_name_from_path(path)

        pairs = pair_range_and_fp_columns(df.columns.tolist())
        if not pairs:
            print(f"[WARN] No (range, fp) pairs found for {path}. Columns: {df.columns.tolist()}")
            continue

        for range_col, fp_col in pairs:
            for _, row in df.iterrows():
                question = row.get("question", "")
                gold_raw = row.get("gold_standard_answer", None)
                range_raw = row.get(range_col, None)
                fp_given = row.get(fp_col, None)

                A = parse_number(gold_raw)
                low, high = parse_range(range_raw)

                if A is None or low is None or high is None:
                    continue
                if fp_given is None or (isinstance(fp_given, float) and math.isnan(fp_given)):
                    continue

                fp_expected = fp_avg_for_range(A, low, high)
                if math.isnan(fp_expected):
                    continue

                diff = abs(float(fp_given) - float(fp_expected))
                if diff > tolerance:
                    bad_rows.append({
                        "LLM": llm,
                        "fp_column": fp_col,
                        "range_column": range_col,
                        "question": question,
                        "gold_standard_answer": gold_raw,
                        "llm_range": range_raw,
                        "fp_given": float(fp_given),
                        "fp_expected": float(fp_expected),
                        "abs_diff": diff,
                    })

    return pd.DataFrame(bad_rows)


if __name__ == "__main__":
    # Put your CSV file paths here:
    csvs = [
        "DeepSeek-Table 1.csv",
        "Gemini-Table 1.csv",
        "ChatGPT-Table 1.csv",
    ]

    # If you run from the same folder as the CSVs, this is enough.
    # Otherwise, replace with full paths.
    incorrect = check_fp_scores(csvs, tolerance=1e-4)

    if incorrect.empty:
        print("All FP scores match the expected values (within tolerance).")
    else:
        print(f"Found {len(incorrect)} incorrect FP scores.")
        print(incorrect[["LLM", "fp_column", "question", "gold_standard_answer", "llm_range", "fp_given", "fp_expected", "abs_diff"]])

        # optional: write results
        incorrect.to_csv("incorrect_fp_scores.csv", index=False)
        print("Wrote: incorrect_fp_scores.csv")