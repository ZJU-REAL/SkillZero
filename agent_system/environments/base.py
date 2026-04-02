# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
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

from typing import List, Tuple, Dict, Union, Any, Optional
import torch
import numpy as np
import os
from agent_system.environments.prompts import *
from collections import defaultdict
from agentocr import OCRTool

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


def parse_highlight_configs(config_value):
    """
    Parse highlight_configs from various formats into a list of dicts.
    
    Supported formats:
    1. None or empty string: returns None
    2. String format: "context1:r,g,b;context2:r,g,b"
       Example: "<search>:0,0,255;</search>:0,0,255;<information>:255,0,0"
    3. List of dicts: [{"context": "...", "color": [r, g, b]}, ...]
    
    Returns:
        List of dicts with 'context' and 'color' keys, or None
    """
    if config_value is None or config_value == '' or config_value == 'null':
        return None
    
    # If it's already a list of dicts, convert colors to tuples
    if isinstance(config_value, (list, tuple)):
        return [
            {
                'context': cfg.get('context', ''),
                'color': tuple(cfg.get('color', [0, 0, 0]))
            }
            for cfg in config_value
        ]
    
    # Parse string format: "context1:r,g,b;context2:r,g,b"
    if isinstance(config_value, str):
        result = []
        # Split by semicolon to get individual highlight configs
        parts = config_value.split(';')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Find the last colon that separates context from color
            # This handles cases where context itself contains colons
            last_colon_idx = part.rfind(':')
            if last_colon_idx == -1:
                continue
            
            context = part[:last_colon_idx]
            color_str = part[last_colon_idx + 1:]
            
            try:
                # Parse color as "r,g,b"
                color = tuple(int(c.strip()) for c in color_str.split(','))
                if len(color) == 3:
                    result.append({
                        'context': context,
                        'color': color
                    })
            except ValueError:
                continue
        
        return result if result else None
    
    return None

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        """
        Initialize the environment manager.
        
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - config: Configuration object.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.config = config
        self.ocr_config = config.ocr

        # Initialize OCRTool if enabled
        use_ocr = getattr(self.ocr_config, 'use_ocr', False)

        # Config keys that should not be passed to OCRTool (handled separately or nested)
        excluded_keys = [
            'use_ocr',  # OCRTool uses 'enabled' instead
            'use_parallel', 'max_workers',  # handled separately
            'compact_mode',  # nested config, handled separately
            'agent_select_compression',  # nested config, environment manager specific
            'highlight_configs',  # handled separately to convert list format
        ]

        if use_ocr:
            # Extract compact_mode settings from nested config
            compact_config = self.ocr_config.get('compact_mode', {})
            compact_mode = compact_config.get('enable', False) if compact_config else False
            compact_symbol = compact_config.get('compact_symbol', '⏎') if compact_config else '⏎'

            # Extract and parse highlight_configs (supports string format for Hydra compatibility)
            # Use environment variable HIGHLIGHT_CONFIGS to avoid Hydra parsing issues with < > characters
            highlight_configs_raw = os.environ.get('HIGHLIGHT_CONFIGS', None)
            if highlight_configs_raw is None:
                highlight_configs_raw = self.ocr_config.get('highlight_configs', None)
            highlight_configs = parse_highlight_configs(highlight_configs_raw)

            self.ocr_tool = OCRTool(
                enabled=True,
                use_parallel=self.ocr_config.get('use_parallel', True),
                max_workers=self.ocr_config.get('max_workers', None),
                compact_mode=compact_mode,
                compact_symbol=compact_symbol,
                highlight_configs=highlight_configs,
                **{k: v for k, v in self.ocr_config.items() if k not in excluded_keys}
            )
        else:
            self.ocr_tool = None


    def reset(self, kwargs) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
        Parameters:
        - kwargs (Dict): Additional keyword arguments for resetting the environment.

        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos
    
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None, # Implement this if needed
            'image': next_obs,
            'anchor': None # For GiGPO only. anchor observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        
        return next_observations, rewards, dones, infos

    def build_text_obs(self,) -> List[str]:
        """
        This function builds the text observation for the agent.
        
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        pass

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return
            
    def save_image(self, image, step):
        """
        Save an image to a file.
        
        Parameters:
        - image (np.ndarray or torch.Tensor): The image to save.
        - path (str): The path to save the image.
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
        
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)
        
        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)