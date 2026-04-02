# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
# Copyright 2026 AgentOCR Team
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

SEARCH_TEMPLATE_NO_HIS = """
{skill_context}You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags.
"""

SEARCH_TEMPLATE = """
{skill_context}You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history, where <search>...</search> wrapped your past search queries and <information>...</information> wrapped the corresponding search results. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags.
"""


#######################################################################################




SEARCH_TEMPLATE_NO_HIS_OCR = """<image>
{skill_context}You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if the image/history already provides sufficient, reliable information to answer with high confidence, provide your final answer within <answer> </answer> tags.
"""

SEARCH_TEMPLATE_OCR = """<image>
{skill_context}You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). 
The image contains the full history:
- Past queries are inside <search>...</search>
- Past results are inside <information>...</information>

Now it's your turn to respond for the current step.
You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if the image/history already provides sufficient, reliable information to answer with high confidence, provide your final answer within <answer> </answer> tags.
"""

SEARCH_COMPRESSION_TEMPLATE_NO_HIS = """
Additionally, select an image compression factor larger than 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. You must output the selected value within <compression> </compression> tags (e.g., <compression>1.1</compression>).
Output format:
1. Reasoning: state what you found in the image.
2. <search>...</search> or <answer>...</answer>
3. <compression>...</compression>
"""

SEARCH_COMPRESSION_TEMPLATE = """
Additionally, select an image compression factor larger than 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. You must output the selected value within <compression> </compression> tags (e.g., <compression>1.1</compression>).
Output format:
1. Reasoning: state what you found in the image.
2. <search>...</search> or <answer>...</answer>
3. <compression>...</compression>
"""