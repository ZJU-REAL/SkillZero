# Copyright 2026 Nanyang Technological University (NTU), Singapore
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

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from PIL import Image


class BaseOCRTool(ABC):
    """
    Base class for OCR tools that convert trajectory history records (text) into images.
    
    This abstract interface allows for flexible implementations while maintaining
    a consistent API for integration with environment managers.
    """
    
    @abstractmethod
    def convert(
        self,
        trajectory_text: Union[str, List[str]],
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Convert trajectory text to image(s).
        
        Args:
            trajectory_text: Single trajectory text string or list of trajectory texts
            **kwargs: Additional configuration parameters
        
        Returns:
            PIL Image object or list of PIL Image objects
        """
        pass
    
    @abstractmethod
    def convert_batch(
        self,
        trajectory_texts: List[str],
        **kwargs
    ) -> List[Image.Image]:
        """
        Convert a batch of trajectory texts to images.
        
        Args:
            trajectory_texts: List of trajectory text strings
            **kwargs: Additional configuration parameters
        
        Returns:
            List of PIL Image objects
        """
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if the OCR tool is enabled and ready to use.
        
        Returns:
            True if the tool is enabled, False otherwise
        """
        pass
