# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Union
import re
import math

def _postprocess_action(action: str) -> str:
    """Trim everything *after* the first closing `</search>` or `</answer>` tag.

    This guards against a common LLM hallucination where an action contains
    several concatenated XML‑like snippets. By hard‑cutting at the first
    relevant close tag we can safely apply non‑greedy regex below.
    """
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def search_projection(actions: List[str], check_compression_tag: bool = False) -> Union[Tuple[List[str], List[int]], Tuple[List[str], List[int], List[float]]]:
    """Project a list of LLM *actions* into (`results`, `valids`, `compression_factors`).

    Extraction logic (order matters):
        1. Grab the **first** complete ``<search>…</search>`` block (case‑insensitive).
        2. If absent, grab the **first** complete ``<answer>…</answer>`` block.
        3. If still absent, store an empty string.

    Validity logic (independent of extraction): ``valids[i]`` flips to **0** when
    the *original* action text satisfies any of:
        1. Contains **both** ``<search>`` and ``<answer>`` tags.
        2. Contains more than one ``<search>`` tag or more than one ``<answer>`` tag.
        3. If compression tag checking is enabled and compression tag is missing or invalid.

    The extracted block (if any) is **not** cleared when a validity rule fails –
    downstream callers can still inspect the fragment while trusting the flag.

    Args:
        actions: List of action strings from LLM
        check_compression_tag: Whether to extract and validate compression factors

    Returns:
        If check_compression_tag is True: (results, valids, compression_factors)
        If check_compression_tag is False: (results, valids)
    """

    results: List[str] = []
    valids: List[int] = [1] * len(actions)
    if check_compression_tag:
        compression_factors: List[float] = [1.0] * len(actions)  # default compression factor

    # --- Pre‑compiled patterns ------------------------------------------------
    re_search_block = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
    re_answer_block = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    re_search_tag = re.compile(r"<search>", re.IGNORECASE)
    re_answer_tag = re.compile(r"<answer>", re.IGNORECASE)

    for i, action in enumerate(actions):
        is_anwser = False
        original_action = action  # Keep untouched for validity checks
        trimmed_action = _postprocess_action(action)

        # --- Extraction -----------------------------------------------------
        m = re_search_block.search(trimmed_action)
        if m:
            results.append(f"<search>{m.group(1).strip()}</search>")
        else:
            m = re_answer_block.search(trimmed_action)
            if m:
                results.append(f"<answer>{m.group(1).strip()}</answer>")
                is_anwser = True
            else:
                results.append("")
                valids[i] = 0

        # --- Validity checks -------------------------------------------------
        n_search = len(re_search_tag.findall(original_action))
        n_answer = len(re_answer_tag.findall(original_action))

        # Both search and answer present
        if n_search and n_answer:
            valids[i] = 0
        # Multiple identical tags
        if n_search > 1 or n_answer > 1:
            valids[i] = 0


        # Extract compression factor from <compression>...</compression>
        if check_compression_tag:
            if is_anwser:
                compression_factors[i] = 1.0
            else:
                comp_start_tag = "<compression>"
                comp_end_tag = "</compression>"
                comp_start_idx = original_action.lower().find(comp_start_tag)
                comp_end_idx = original_action.lower().find(comp_end_tag)
                
                if comp_start_idx != -1 and comp_end_idx != -1:
                    try:
                        compression_str = original_action[comp_start_idx + len(comp_start_tag):comp_end_idx].strip()
                        compression_value = float(compression_str)
                        # Clamp to [1.0, 5.0] (higher values = more compression)
                        if math.isnan(compression_value) or not math.isfinite(compression_value):
                            compression_value = 1.0
                        elif compression_value < 1.0:
                            compression_value = 1.0
                        elif compression_value > 5.0:
                            compression_value = 5.0
                        compression_factors[i] = compression_value
                    except:
                        # If parsing fails, default to max compression
                        compression_factors[i] = 1.0
                else:
                    compression_factors[i] = 1.0

    if check_compression_tag:
        return results, valids, compression_factors
    else:
        return results, valids
