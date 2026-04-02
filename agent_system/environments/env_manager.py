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

from typing import List, Tuple, Dict, Union, Any, Optional
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import re
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf
import time


GAMEFILE_TO_SKILL_KEY = {
    "pick_and_place": "TASK: pick_and_place",
    "pick_two_obj_and_place": "TASK: pick_and_place",
    "look_at_obj_in_light": "TASK: look_at_obj_in_light",
    "pick_heat_then_place_in_recep": "TASK: heat",
    "pick_cool_then_place_in_recep": "TASK: cool",
    "pick_clean_then_place_in_recep": "TASK: clean",
}


def load_skill_file(filepath):
    """Load a skill file with section markers like '### SECTION_NAME ###'."""
    with open(filepath, 'r') as f:
        content = f.read()
    sections = {}
    parts = re.split(r'### (.+?) ###\s*\n', content)
    for i in range(1, len(parts), 2):
        key = parts[i].strip()
        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[key] = value
    return sections


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)
        self.agent_select_compression_enable = self.ocr_config.agent_select_compression.get('enable', False)

        # Skill support (env-level config, shared across envs)
        curriculum_cfg = config.env.get('curriculum_learning', {})
        curriculum_enabled = curriculum_cfg.get('enable', False)
        self.use_skill = config.env.get('use_skill', False) or curriculum_enabled
        self.skills = None
        self.task_to_sections: Dict[str, List[str]] = {}
        self.skill_types: List[Optional[str]] = []

        if curriculum_enabled:
            print(
                "[SearchEnvironmentManager] Curriculum learning enabled, "
                "skills will be set by CurriculumSkillManager"
            )
        elif self.use_skill:
            skill_file = config.env.get('skill_file', None)
            if skill_file is not None:
                self.skills = load_skill_file(skill_file)
                print(f"[SearchEnvironmentManager] Loaded skills from {skill_file}, "
                      f"sections: {list(self.skills.keys())}")
            else:
                print("[SearchEnvironmentManager] use_skill=True but no skill_file specified, skills disabled")

        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.template_no_his = SEARCH_TEMPLATE_NO_HIS_OCR
            self.template = SEARCH_TEMPLATE_OCR
            if self.agent_select_compression_enable:
                self.template_no_his += SEARCH_COMPRESSION_TEMPLATE_NO_HIS
                self.template += SEARCH_COMPRESSION_TEMPLATE
        else:
            self.template_no_his = SEARCH_TEMPLATE_NO_HIS
            self.template = SEARCH_TEMPLATE

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.skill_types = []
        for info in infos:
            if isinstance(info, dict):
                self.skill_types.append(info.get("skill_type"))
            else:
                self.skill_types.append(None)

        self.memory.reset(batch_size=len(obs))
        self.active_masks = [True] * len(obs)

        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.ocr_time = 0
            # Reset OCRTool to clear all caches and statistics
            self.ocr_tool.reset()

        full_text_obs, trajectory_images = self.build_text_obs(obs, init=True)

        observations = {
            "text": full_text_obs,
            "image": trajectory_images,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        # Extract actions, validity, and compression factors from LLM responses
        if self.ocr_tool and self.ocr_tool.is_enabled() and self.agent_select_compression_enable:
            actions, valids, compression_factors = self.projection_f(text_actions, check_compression_tag=True)
        else:
            actions, valids = self.projection_f(text_actions)
            compression_factors = None
        
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })
        
        for i, done in enumerate(dones):
            if done:
                self.active_masks[i] = False

        full_text_obs, trajectory_images = self.build_text_obs(next_obs, compression_factors=compression_factors)
        next_observations = {
            "text": full_text_obs,
            "image": trajectory_images,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            if compression_factors is not None:
                info['compression_factor'] = compression_factors[i] if self.active_masks[i] else 1.0

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        compression_factors: Optional[List[float]] = None,
        init: bool = False
    ) -> Tuple[List[str], Optional[List]]:
        """
        This function builds the text observation for the agent and optionally renders trajectory history as images.
        
        Returns:
            Tuple of (text_observations, trajectory_images):
            - text_observations: List of processed text observations
            - trajectory_images: List of PIL Images (or None if OCR is disabled/not available)
        """
        postprocess_text_obs = []
        trajectory_images = None
        memory_contexts, valid_lens = None, None
        
        # Fetch memory contexts if needed (for both OCR and non-OCR cases)
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )
        
        # If OCRTool is enabled, generate images (blank for init, or from history)
        if self.ocr_tool and self.ocr_tool.is_enabled():
            start_time = time.time()

            # Get step count from memory (use first env's memory length as reference)
            step_info = str(len(self.memory[0])) if len(self.memory) > 0 else "0"
            
            # Use compression factors chosen by the LLM (per environment)
            # Pass individual compression factors as a list for per-image compression
            if compression_factors is None:
                # Default to no compression (1.0) for all images
                compression_factors = [1.0] * len(text_obs)
            
            # Use use_precise=False for faster processing (significant speedup)
            trajectory_images = self.ocr_tool.convert_texts_to_images(
                memory_contexts, 
                batch_size=len(text_obs),
                active_masks=self.active_masks,
                compression_factor=compression_factors, 
                save_img=False, 
                step_info=step_info,
                use_precise=False,
                enable_cache=True,
                current_steps=[len(self.memory[i]) for i in range(len(text_obs))]
            )
            end_time = time.time()
            self.ocr_time += end_time - start_time
            # print(f"Step {len(self.memory[0])+1}, OCR time: {end_time - start_time}")

        for i in range(len(text_obs)):
            skill_ctx = self._get_skill_context(i)
            if init or self.config.env.history_length <= 0:
                obs_i = self.template_no_his.format(
                    task_description=self.tasks[i],
                    skill_context=skill_ctx,
                )
            else:
                obs_i = self.template.format(
                    task_description=self.tasks[i],
                    memory_context=memory_contexts[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i] if valid_lens else len(self.memory[i]),
                    compression_factor=compression_factors[i] if compression_factors is not None else 1.0,
                    skill_context=skill_ctx,
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs, trajectory_images

    def _get_skill_context(self, env_idx):
        """Build skill context string: general skills plus per-task sections when configured."""
        if not self.use_skill or self.skills is None:
            return ""
        parts = []
        general = self.skills.get("GENERAL SKILLS", "")
        if general:
            parts.append(general)
        if self.skill_types and env_idx < len(self.skill_types):
            task_id = self.skill_types[env_idx]
            if task_id and self.task_to_sections:
                for section_key in self.task_to_sections.get(task_id, []):
                    content = self.skills.get(section_key, "")
                    if content:
                        parts.append(content)
        if parts:
            return "\n\n".join(parts) + "\n\n"
        return ""

    def update_skills(self, new_skills: dict):
        """Dynamically update the active skill set during training."""
        self.skills = new_skills

    def set_use_skill(self, enabled: bool):
        """Toggle skill usage (for w/ vs w/o skill validation)."""
        self.use_skill = enabled

    def set_task_to_sections(self, task_to_sections: dict):
        """Store mapping from task id (e.g. skill_type) to section keys for _get_skill_context."""
        self.task_to_sections = task_to_sections or {}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)

                skill_type = info.get("skill_type")
                if skill_type:
                    success[f"{skill_type}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        self.agent_select_compression_enable = self.ocr_config.agent_select_compression.get('enable', False)

        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.template_no_his = ALFWORLD_TEMPLATE_NO_HIS_OCR
            self.template = ALFWORLD_TEMPLATE_OCR
            if self.agent_select_compression_enable:
                self.template_no_his += ALFWORLD_COMPRESSION_TEMPLATE_NO_HIS
                self.template += ALFWORLD_COMPRESSION_TEMPLATE
        else:
            self.template_no_his = ALFWORLD_TEMPLATE_NO_HIS
            self.template = ALFWORLD_TEMPLATE

        # Skill support (env-level config, with alfworld fallback for backward compatibility)
        alfworld_config = config.env.get('alfworld', {})
        curriculum_cfg = config.env.get('curriculum_learning', {})
        curriculum_enabled = curriculum_cfg.get('enable', False)
        self.use_skill = (
            config.env.get('use_skill', alfworld_config.get('use_skill', False))
            or curriculum_enabled
        )
        self.skills = None
        if curriculum_enabled:
            # Curriculum learning: skills are set via update_skills() by make_envs/trainer
            # Initial skills will be injected by make_envs after creation
            print("[AlfWorldEnvironmentManager] Curriculum learning enabled, skills will be set by CurriculumSkillManager")
        elif self.use_skill:
            skill_file = config.env.get('skill_file', alfworld_config.get('skill_file', None))
            if skill_file is not None:
                self.skills = load_skill_file(skill_file)
                print(f"[AlfWorldEnvironmentManager] Loaded skills from {skill_file}, "
                      f"sections: {list(self.skills.keys())}")
            else:
                print("[AlfWorldEnvironmentManager] use_skill=True but no skill_file specified, skills disabled")
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        self.active_masks = [True] * len(text_obs)
        
        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.ocr_time = 0
            # Reset OCRTool to clear all caches and statistics
            self.ocr_tool.reset()

        full_text_obs, trajectory_images = self.build_text_obs(text_obs, self.envs.get_admissible_commands, compression_factors=None, init=True)
        return {'text': full_text_obs, 'image': trajectory_images, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        # Extract actions, validity, and compression factors from LLM responses
        if self.ocr_tool and self.ocr_tool.is_enabled() and self.agent_select_compression_enable:
            actions, valids, compression_factors = self.projection_f(text_actions, self.envs.get_admissible_commands, check_compression_tag=True)
        else:
            actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
            compression_factors = None
        
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        for i, done in enumerate(dones):
            if done:
                self.active_masks[i] = False

        rewards = to_numpy(rewards) 

        full_text_obs, trajectory_images = self.build_text_obs(text_obs, self.envs.get_admissible_commands, compression_factors=compression_factors)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid and compression_factor to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            if compression_factors is not None:
                info['compression_factor'] = compression_factors[i] if self.active_masks[i] else 1.0

        next_observations = {'text': full_text_obs, 'image': trajectory_images, 'anchor': text_obs}
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
    def _detect_task_type(self, gamefile):
        """Detect ALFWorld task type from gamefile path and return the skill section key."""
        if gamefile is None:
            return None
        for task_key, skill_key in GAMEFILE_TO_SKILL_KEY.items():
            if task_key in gamefile:
                return skill_key
        return None

    def _get_skill_context(self, env_idx):
        """Build skill context string for the given environment index."""
        if not self.use_skill or self.skills is None:
            return ""

        parts = []
        general = self.skills.get("GENERAL SKILLS", "")
        if general:
            parts.append(general)

        task_skill_key = self._detect_task_type(self.gamefile[env_idx])
        if task_skill_key:
            task_skills = self.skills.get(task_skill_key, "")
            if task_skills:
                parts.append(task_skills)

        if parts:
            return "\n\n".join(parts) + "\n\n"
        return ""

    def update_skills(self, new_skills: dict):
        """Dynamically update the active skill set during training."""
        self.skills = new_skills

    def set_use_skill(self, enabled: bool):
        """Toggle skill usage (for w/ vs w/o skill validation)."""
        self.use_skill = enabled

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], compression_factors: Optional[List[float]] = None, init: bool = False) -> Tuple[List[str], Optional[List]]:
        """
        This function builds the text observation for the agent and optionally renders trajectory history as images.
        
        Returns:
            Tuple of (text_observations, trajectory_images):
            - text_observations: List of processed text observations
            - trajectory_images: List of PIL Images (or None if OCR is disabled/not available)
        """
        postprocess_text_obs = []
        trajectory_images = None
        
        # If OCRTool is enabled, generate images (blank for init, or from history)
        if self.ocr_tool and self.ocr_tool.is_enabled():
            start_time = time.time()
            if init or self.config.env.history_length <= 0:
                memory_contexts, valid_lens = None, None
            else:
                memory_contexts, valid_lens = self.memory.fetch(self.config.env.history_length, obs_key="text_obs", action_key="action")

            # Get step count from memory (use first env's memory length as reference)
            step_info = str(len(self.memory[0])) if len(self.memory) > 0 else "0"
            
            # Use compression factors chosen by the LLM (per environment)
            # Pass individual compression factors as a list for per-image compression
            if compression_factors is None:
                # Default to no compression (1.0) for all images
                compression_factors = [1.0] * len(text_obs)
            
            # Use use_precise=False for faster processing (significant speedup)
            trajectory_images = self.ocr_tool.convert_texts_to_images(
                memory_contexts, 
                batch_size=len(text_obs), 
                active_masks=self.active_masks,
                compression_factor=compression_factors, 
                save_img=False, 
                step_info=step_info,
                use_precise=False,
                enable_cache=True,
                current_steps=[len(self.memory[i]) for i in range(len(text_obs))]
            )
            end_time = time.time()
            self.ocr_time += end_time - start_time
            # print(f"Step {len(self.memory[0])+1}, OCR time: {end_time - start_time}")
        elif not init and self.config.env.history_length > 0:
            # OCRTool not enabled, but we still need to fetch memory for text obs
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')
            skill_ctx = self._get_skill_context(i)
            if init or self.config.env.history_length <= 0:
                obs = self.template_no_his.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    skill_context=skill_ctx,
                )
            else:
                obs = self.template.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    compression_factor=compression_factors[i] if compression_factors is not None else 1.0,
                    skill_context=skill_ctx,
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs, trajectory_images

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


def _resolve_curriculum_total_steps(config):
    """Prefer explicit total_training_steps; else total_epochs; else default."""
    tts = config.trainer.get("total_training_steps", None)
    if tts is not None:
        return int(tts)
    te = config.trainer.get("total_epochs", None)
    if te is not None:
        return int(te)
    return 150


def _maybe_create_curriculum(config, envs, val_envs, test_envs=None):
    """If curriculum_learning is enabled, build CurriculumSkillManager and inject skills."""
    curriculum_cfg = config.env.get("curriculum_learning", {})
    if not curriculum_cfg.get("enable", False):
        return None
    mapping_path = curriculum_cfg.get("skill_mapping_file")
    if not mapping_path:
        print("[make_envs] curriculum_learning.enable=True but skill_mapping_file is not set")
        return None

    from agent_system.environments.curriculum_skill_manager import CurriculumSkillManager

    mapping_path = os.path.expanduser(mapping_path)
    total_steps = _resolve_curriculum_total_steps(config)
    test_freq = int(config.trainer.get("test_freq", 10))
    if test_freq <= 0:
        test_freq = 10

    curriculum_manager = CurriculumSkillManager(
        skill_mapping_file=mapping_path,
        max_set_schedule=list(curriculum_cfg.get("max_set_schedule", [6])),
        total_steps=total_steps,
        test_freq=test_freq,
    )
    initial_skills = curriculum_manager.get_active_skills()
    task_to_sections = curriculum_manager.get_task_to_sections()
    for mgr in (envs, val_envs):
        mgr.update_skills(initial_skills)
        if hasattr(mgr, "set_task_to_sections"):
            mgr.set_task_to_sections(task_to_sections)
    if test_envs is not None:
        test_envs.update_skills(initial_skills)
        if hasattr(test_envs, "set_task_to_sections"):
            test_envs.set_task_to_sections(task_to_sections)
    print(
        f"[make_envs] Curriculum learning: initial skills "
        f"{curriculum_manager.get_active_skill_names()}"
    )
    return curriculum_manager


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)

        test_envs = None
        test_files = config.data.get("test_files", None)
        if test_files is not None:
            _test_envs = build_search_envs(
                seed=config.env.seed + 2000,
                env_num=config.data.val_batch_size,
                group_n=1,
                is_train=False,
                env_config=config.env,
            )
            test_envs = SearchEnvironmentManager(_test_envs, projection_f, config)

        curriculum_manager = _maybe_create_curriculum(config, envs, val_envs, test_envs)
        return envs, val_envs, curriculum_manager, test_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs, None, None
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)

        curriculum_manager = _maybe_create_curriculum(config, envs, val_envs, None)
        return envs, val_envs, curriculum_manager, None
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs, None, None
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs, None, None
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs, None, None
    else:
        print("Environment not supported")
        exit(1)