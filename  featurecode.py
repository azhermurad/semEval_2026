import re
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy




def get_indent_levels(lines):
    indents = []
    for line in lines:
        if line.strip():
            indents.append(len(line) - len(line.lstrip()))
    return indents


def token_shape(token):
    if re.fullmatch(r"[A-Z_]+", token):
        return "UPPER_SNAKE"
    if re.fullmatch(r"[a-z]+_[a-z_]+", token):
        return "snake_case"
    if re.fullmatch(r"[a-z]+[A-Z][a-zA-Z]*", token):
        return "camelCase"
    if re.fullmatch(r"[a-zA-Z]+\d+", token):
        return "alpha_num"
    if len(token) > 10:
        return "long_token"
    return "other"







def extract_features(code: str):
    lines = code.split("\n")
    total_lines = len(lines)

    if total_lines == 0:
        return {}

    line_lengths = [len(l) for l in lines if l.strip()]
    blank_lines = [l for l in lines if not l.strip()]

    # --- Comments (language-agnostic heuristic)
    comment_lines = [
        l for l in lines if l.strip().startswith(("#", "//", "/*", "*"))
    ]

    # --- Structural features
    features = {
        "num_lines": total_lines,
        "avg_line_length": np.mean(line_lengths) if line_lengths else 0,
        "std_line_length": np.std(line_lengths) if line_lengths else 0,
        "blank_line_ratio": len(blank_lines) / total_lines,
        "comment_line_ratio": len(comment_lines) / total_lines,
        "code_char_ratio": sum(len(l) for l in lines) / (len(code) + 1e-6),
    }

    # --- Indentation
    indent_levels = get_indent_levels(lines)
    if indent_levels:
        indent_counts = Counter(indent_levels)
        features.update({
            "avg_indent": np.mean(indent_levels),
            "indent_std": np.std(indent_levels),
            "indent_entropy": entropy(list(indent_counts.values()))
        })
    else:
        features.update({
            "avg_indent": 0,
            "indent_std": 0,
            "indent_entropy": 0
        })

    # --- Tabs vs spaces
    tab_count = sum("\t" in l for l in lines)
    space_indent_count = sum(l.startswith(" ") for l in lines)
    features["tabs_vs_spaces_ratio"] = tab_count / (space_indent_count + 1e-6)

    # --- Token shapes
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code)
    shapes = [token_shape(t) for t in tokens]
    shape_counts = Counter(shapes)

    for shape, count in shape_counts.items():
        features[f"shape_{shape}"] = count / (len(tokens) + 1e-6)

    # --- Repetition
    unique_lines = set(lines)
    features["repeated_line_ratio"] = 1 - len(unique_lines) / total_lines

    token_counts = Counter(tokens)
    repeated_tokens = sum(c for c in token_counts.values() if c > 1)
    features["repeated_token_ratio"] = repeated_tokens / (len(tokens) + 1e-6)

    # --- Comment semantics
    comment_lengths = [len(l) for l in comment_lines]
    features["avg_comment_length"] = np.mean(comment_lengths) if comment_lengths else 0

    full_sentence_comments = [
        l for l in comment_lines if l.strip().endswith(".")
    ]
    features["full_sentence_comment_ratio"] = (
        len(full_sentence_comments) / (len(comment_lines) + 1e-6)
    )

    return features




# df is your dataset
feature_df = df["code"].apply(extract_features).apply(pd.Series)

# Final training table
X = feature_df.fillna(0)
y = df["label"]
