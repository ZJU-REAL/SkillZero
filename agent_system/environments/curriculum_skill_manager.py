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

"""
Curriculum Skill Manager for dynamic skill selection during ALFWorld training.
"""

import json
import os
from typing import Dict, List, Optional

from agent_system.environments.env_manager import load_skill_file


class CurriculumSkillManager:
    """
    Manages curriculum-based skill selection for ALFWorld training.
    Dynamically updates active skills based on validation delta (with_skill - without_skill).
    """

    def __init__(
        self,
        skill_mapping_file: str,
        max_set_schedule: List[int],
        total_steps: int,
        test_freq: int,
    ):
        """
        Args:
            skill_mapping_file: Path to JSON with skill_files and task_to_skill.
            max_set_schedule: [6, 3, 0] - max active skill categories per phase
            total_steps: total training steps (e.g., 150)
            test_freq: validation frequency (e.g., 10)
        """
        self.skill_mapping_file = os.path.abspath(skill_mapping_file)
        self._mapping_dir = os.path.dirname(self.skill_mapping_file)

        with open(self.skill_mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self.task_to_skill: Dict[str, str] = dict(mapping.get("task_to_skill", {}))
        skill_files_cfg: Dict[str, str] = dict(mapping.get("skill_files", {}))

        self.skill_files: Dict[str, Optional[str]] = {}
        for key, rel_path in skill_files_cfg.items():
            self.skill_files[key] = os.path.join(self._mapping_dir, rel_path)

        self.max_set_schedule = max_set_schedule
        self.total_steps = total_steps
        self.test_freq = test_freq

        # Validate schedule: len(max_set_schedule) must divide total_steps / test_freq
        num_validations = total_steps // test_freq
        assert num_validations % len(max_set_schedule) == 0, (
            f"len(max_set_schedule)={len(max_set_schedule)} must divide "
            f"total_steps/test_freq={num_validations}"
        )

        # Load all skill content from files
        self._all_skills: Dict[str, Dict[str, str]] = {}
        for key, path in self.skill_files.items():
            if path is not None and os.path.isfile(path):
                self._all_skills[key] = load_skill_file(path)
            else:
                self._all_skills[key] = {}

        # Build merged skills dict: section_key -> content (for env_manager compatibility)
        self._active_skill_names: List[str] = list(self.skill_files.keys())
        self._active_skills: Dict[str, str] = self._build_skills_from_names(
            self._active_skill_names
        )

    def _build_skills_from_names(
        self, active_names: List[str]
    ) -> Dict[str, str]:
        """Build merged skills dict {section_key: content} from active skill file names."""
        merged: Dict[str, str] = {}
        for name in active_names:
            if name not in self._all_skills:
                continue
            for sec, content in self._all_skills[name].items():
                merged[sec] = content
        return merged

    def get_task_to_sections(self) -> Dict[str, List[str]]:
        """Map each task id to section keys from that task's skill file."""
        out: Dict[str, List[str]] = {}
        for task, skill_name in self.task_to_skill.items():
            if skill_name in self._all_skills:
                out[task] = list(self._all_skills[skill_name].keys())
        return out

    def get_current_max_set(self, global_step: int) -> int:
        """Return max_set for the current phase."""
        num_validations = self.total_steps // self.test_freq
        validations_per_phase = num_validations // len(self.max_set_schedule)
        # Which validation are we at (0-indexed)?
        validation_idx = (global_step - 1) // self.test_freq
        phase_idx = min(
            validation_idx // validations_per_phase,
            len(self.max_set_schedule) - 1,
        )
        return self.max_set_schedule[phase_idx]

    def update_skill_set(
        self, delta_success_rates: Dict[str, float], max_set: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Given per-task delta success rates, update active skills.
        - Remove skills where delta < 0
        - Sort remaining by delta descending
        - Keep top max_set (general_skills always retained unless max_set=0)
        Returns: updated active skills dict {section_key: content} for env_manager

        Args:
            delta_success_rates: per-task delta (with_skill - without_skill), e.g.
                {"pick_and_place_success_rate": 0.1, "pick_two_obj_and_place_success_rate": -0.05, ...}
            max_set: max number of active skill categories; if None, use first schedule value
        """
        if max_set is None:
            max_set = self.max_set_schedule[0] if self.max_set_schedule else 6

        # Map task-level deltas to skill-file-level deltas
        # Keys in delta_success_rates are like "pick_and_place_success_rate", "success_rate", etc.
        # Extract task name: "pick_and_place_success_rate" -> "pick_and_place"
        skill_deltas: Dict[str, List[float]] = {}
        for key, delta in delta_success_rates.items():
            if key == "success_rate" or "success_rate" not in key:
                continue
            task = key.replace("_success_rate", "")
            skill_key = self.task_to_skill.get(task)
            if skill_key is None:
                continue
            if skill_key not in skill_deltas:
                skill_deltas[skill_key] = []
            skill_deltas[skill_key].append(delta)

        # Aggregate: for pick_and_place, average both task types
        skill_deltas_agg: Dict[str, float] = {
            k: sum(v) / len(v) for k, v in skill_deltas.items()
        }

        # Filter: remove skills with delta < 0
        candidates = [
            (k, v)
            for k, v in skill_deltas_agg.items()
            if v >= 0 and k != "general_skills"
        ]
        # Sort by delta descending
        candidates.sort(key=lambda x: -x[1])

        # Build active set: general_skills + top (max_set - 1) task skills
        active_names = ["general_skills"]
        if max_set > 0:
            for i, (name, _) in enumerate(candidates):
                if i >= max_set - 1:  # -1 because general_skills already in
                    break
                active_names.append(name)
        else:
            # max_set=0: no skills at all
            active_names = []

        self._active_skill_names = active_names
        self._active_skills = self._build_skills_from_names(active_names)
        return self._active_skills

    def get_active_skills(self) -> Dict[str, str]:
        """Return current active skills dict for env manager (section_key -> content)."""
        return self._active_skills.copy()

    def get_full_skills(self) -> Dict[str, str]:
        """Merged skills from every skill file (ignores curriculum pruning)."""
        return self._build_skills_from_names(list(self.skill_files.keys()))

    def get_active_skill_names(self) -> List[str]:
        """Return names of currently active skill files (for logging)."""
        return self._active_skill_names.copy()
