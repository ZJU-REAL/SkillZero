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

from typing import List, Tuple
import re
import math

def alfworld_projection(actions: List[str], action_pools: List[List[str]], check_compression_tag: bool = False) -> Tuple[List[str], List[int], List[float]]:
    """
    An function to process the actions and compression factors
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    check_compression_tag: whether to check the compression tag, default is False.

    Returns:
        actions: List of extracted actions
        valids: List of validity flags (0 or 1)
        compression_factors: List of compression factors (default 1.0 if not specified)
    """

    valids = [0] * len(actions)
    if check_compression_tag:
        compression_factors = [1.0] * len(actions)  # default compression factor

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Check that each tag appears at most once
        action_start_count = original_str.lower().count("<action>")
        action_end_count = original_str.lower().count("</action>")
        if action_start_count > 1 or action_end_count > 1:
            valids[i] = 0
            actions[i] = ""
            if check_compression_tag:
                compression_factors[i] = 1.0
            continue

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = "" 
            else:
                # Extract just the content between the tags
                extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
                
                actions[i] = extracted_action
                valids[i] = 1

        except:
            actions[i] = ""

        # Extract compression factor from <compression>...</compression>
        if check_compression_tag:
            comp_start_tag = "<compression>"
            comp_end_tag = "</compression>"
            comp_start_count = original_str.lower().count(comp_start_tag)
            comp_end_count = original_str.lower().count(comp_end_tag)
            if comp_start_count > 1 or comp_end_count > 1:
                valids[i] = 0
            
            comp_start_idx = original_str.lower().find(comp_start_tag)
            comp_end_idx = original_str.lower().find(comp_end_tag)
            
            if comp_start_idx != -1 and comp_end_idx != -1:
                try:
                    compression_str = original_str[comp_start_idx + len(comp_start_tag):comp_end_idx].strip()
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

        # check <think>...</think>
        think_start_tag = "<think>"
        think_end_tag = "</think>"
        think_start_count = original_str.count(think_start_tag)
        think_end_count = original_str.count(think_end_tag)
        if think_start_count > 1 or think_end_count > 1:
            valids[i] = 0
        
        think_start_idx = original_str.find(think_start_tag)
        think_end_idx = original_str.find(think_end_tag)
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    if check_compression_tag:
        return actions, valids, compression_factors
    else:
        return actions, valids
