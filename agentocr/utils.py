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

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict, Any
import re
from functools import lru_cache

# Global font cache
_FONT_CACHE = {}

# Compact mode special symbol
COMPACT_NEWLINE_SYMBOL = "⏎"  # Return symbol to represent newlines


def apply_compact_mode(text: str, symbol: str = COMPACT_NEWLINE_SYMBOL) -> str:
    """
    Convert text to compact mode by replacing newlines with a special symbol.
    
    Args:
        text: Original text with newlines
        symbol: Symbol to replace newlines with (default: ⏎)
    
    Returns:
        Text with newlines replaced by the symbol, treated as single paragraph
    """
    if not text:
        return ""
    
    # Replace newlines with the symbol (add space around for readability)
    compact_text = text.replace('\n', f' {symbol} ')
    # Clean up multiple spaces
    compact_text = ' '.join(compact_text.split())
    return compact_text


def get_compact_symbol_positions(text: str, symbol: str = COMPACT_NEWLINE_SYMBOL) -> List[int]:
    """
    Find positions of compact mode symbols in text.
    
    Args:
        text: Text containing compact symbols
        symbol: The symbol to find
    
    Returns:
        List of character positions where the symbol appears
    """
    positions = []
    start = 0
    while True:
        pos = text.find(symbol, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

def preprocess_trajectory_contexts(trajectory_contexts: List[str]) -> List[str]:
    """
    Preprocess trajectory contexts.
    """
    # replace \" with "
    return [context.replace('\\"', '\"') for context in trajectory_contexts]

def _get_cached_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    """
    Get or create a cached font object to avoid repeated font loading.
    This provides significant speedup for repeated text rendering.
    """
    cache_key = (font_path or "default", font_size)
    
    if cache_key not in _FONT_CACHE:
        font = None
        font_paths = []
        
        if font_path:
            font_paths.append(font_path)
        
        # Prioritize monospace fonts for better packing efficiency
        font_paths.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Monaco.ttf",  # macOS
            "C:\\Windows\\Fonts\\consola.ttf",   # Windows
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "Arial.ttf",
        ])
        
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        _FONT_CACHE[cache_key] = font
    
    return _FONT_CACHE[cache_key]


# @lru_cache(maxsize=128)
# def parse_trajectory_text(text: str) -> Tuple[Tuple[str, str, str], ...]:
#     """
#     Extract Observation and Action pairs from trajectory text.

#     Args:
#         text: The original trajectory text, can contain multiple rounds

#     Returns:
#         A tuple of tuples in the format ((obs_num, obs_text, action_text), ...)
#         (Changed to tuple for caching compatibility)
#     """
#     pairs = []
    
#     # Match [Observation N: '...', Action N: '...'] format
#     pattern = r"\[Observation\s+(\d+):\s*'(.*?)',\s*Action\s+\d+:\s*'(.*?)'\]"
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     for match in matches:
#         obs_num, obs_text, action_text = match
#         # Unescape the text (handle \n, \t, etc.)
#         obs_text = obs_text.replace('\\n', '\n').replace('\\t', '\t')
#         action_text = action_text.replace('\\n', '\n').replace('\\t', '\t')
        
#         # Replace all newlines with spaces to keep text on single line
#         obs_text = obs_text.replace('\n', ' ').replace('\r', ' ')
#         action_text = action_text.replace('\n', ' ').replace('\r', ' ')
        
#         # Remove multiple consecutive spaces
#         obs_text = ' '.join(obs_text.split())
#         action_text = ' '.join(action_text.split())
        
#         pairs.append((obs_num, obs_text, action_text))
    
#     return tuple(pairs)


# def format_trajectory_compact(pairs: List[Tuple[str, str, str]]) -> str:
#     """
#     Format Observation-Action pairs into a compact format without empty lines
#     """
#     lines = []
#     for obs_num, obs_text, action_text in pairs:
#         lines.append(f"[Observation {obs_num}]: {obs_text}")
#         lines.append(f"[Action {obs_num}]: {action_text}")
    
#     result = "\n".join(lines)
#     return result


def wrap_text_fast(text: str, max_chars_per_line: int) -> List[Tuple[str, bool]]:
    """
    Fast text wrapping based on character count.
    Returns a list of tuples (line_text, is_paragraph_end) to track paragraph boundaries.
    Optimized for speed with early returns and minimal operations.
    """
    if not text:
        return []
    
    lines = []
    paragraphs = text.split('\n')
    num_paragraphs = len(paragraphs)
    
    for para_idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append(("", True))
            continue
        
        # Fast path for short paragraphs
        if len(paragraph) <= max_chars_per_line:
            is_last_para = para_idx == num_paragraphs - 1
            lines.append((paragraph, not is_last_para))
            continue
        
        words = paragraph.split()
        current_line = ""
        current_len = 0
        
        for word in words:
            word_len = len(word)
            test_len = current_len + (1 if current_len else 0) + word_len
            
            if test_len <= max_chars_per_line:
                if current_line:
                    current_line += " " + word
                    current_len = test_len
                else:
                    current_line = word
                    current_len = word_len
            else:
                if current_line:
                    lines.append((current_line, False))
                
                if word_len > max_chars_per_line:
                    # Split long words
                    for i in range(0, word_len, max_chars_per_line):
                        lines.append((word[i:i + max_chars_per_line], False))
                    current_line = ""
                    current_len = 0
                else:
                    current_line = word
                    current_len = word_len
        
        if current_line:
            is_last_para = para_idx == num_paragraphs - 1
            lines.append((current_line, not is_last_para))
    
    return lines


def wrap_text_compact(
    text: str, 
    max_chars_per_line: int
) -> Tuple[List[Tuple[str, bool]], int, str]:
    """
    Wrap text for compact mode (single paragraph, no newline splitting).
    
    This function is specifically designed for compact mode caching:
    - All text is treated as a single paragraph
    - Returns information about complete vs incomplete lines for caching
    
    Args:
        text: Text to wrap (should already have newlines replaced with symbols)
        max_chars_per_line: Maximum characters per line
    
    Returns:
        Tuple of:
        - lines: List of (line_text, is_complete) tuples
                 is_complete=True means the line filled the available width
        - complete_char_count: Number of characters in complete lines
        - incomplete_text: Text that doesn't fill a complete line (for next render)
    """
    if not text:
        return [], 0, ""
    
    text = text.strip()
    if not text:
        return [], 0, ""
    
    lines = []
    words = text.split()
    current_line = ""
    current_len = 0
    char_position = 0  # Track position in original text
    complete_char_count = 0
    
    for word_idx, word in enumerate(words):
        word_len = len(word)
        test_len = current_len + (1 if current_len else 0) + word_len
        
        if test_len <= max_chars_per_line:
            if current_line:
                current_line += " " + word
                current_len = test_len
            else:
                current_line = word
                current_len = word_len
        else:
            if current_line:
                # This line is complete (filled to capacity)
                lines.append((current_line, True))
                complete_char_count += len(current_line) + 1  # +1 for the space that would follow
            
            if word_len > max_chars_per_line:
                # Split long words
                for i in range(0, word_len, max_chars_per_line):
                    chunk = word[i:i + max_chars_per_line]
                    is_complete = (i + max_chars_per_line < word_len)
                    lines.append((chunk, is_complete))
                    if is_complete:
                        complete_char_count += len(chunk)
                current_line = ""
                current_len = 0
            else:
                current_line = word
                current_len = word_len
    
    # Handle the last line (incomplete - didn't fill the width)
    incomplete_text = ""
    if current_line:
        lines.append((current_line, False))  # Last line is never "complete"
        # The incomplete text is the last line's content
        incomplete_text = current_line
    
    return lines, complete_char_count, incomplete_text


def wrap_text_precise(text: str, max_width: int, font, font_size: int) -> List[Tuple[str, bool]]:
    """
    Precise text wrapping using actual font measurements for optimal packing.
    Returns a list of tuples (line_text, is_paragraph_end) to track paragraph boundaries.
    """
    lines = []
    paragraphs = text.split('\n')
    
    for para_idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append(("", True))
            continue
        
        words = paragraph.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            
            # Use actual font measurement
            try:
                bbox = font.getbbox(test_line)
                text_width = bbox[2] - bbox[0]
            except:
                # Fallback to character-based estimation
                text_width = len(test_line) * font_size * 0.6
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append((current_line, False))
                current_line = word
        
        if current_line:
            # Mark the last line of each paragraph
            is_last_para = para_idx == len(paragraphs) - 1
            lines.append((current_line, not is_last_para))
    
    return lines


# Cache for font metrics to avoid repeated calculations
_FONT_METRICS_CACHE = {}

def get_font_metrics(font, font_size: int) -> Tuple[float, int]:
    """
    Get accurate font metrics for optimal layout calculation.
    Returns (average_char_width, line_height)
    
    Optimized for maximum density: minimal line spacing while maintaining readability.
    Cached for performance.
    """
    # Use font object id and size as cache key
    cache_key = (id(font), font_size)
    
    if cache_key in _FONT_METRICS_CACHE:
        return _FONT_METRICS_CACHE[cache_key]
    
    # Test with a representative set of characters
    sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?()[]{}@#$%^&*-_=+/\\"
    
    try:
        bbox = font.getbbox(sample_text)
        total_width = bbox[2] - bbox[0]
        avg_char_width = total_width / len(sample_text)
        line_height = bbox[3] - bbox[1]
        # Ultra-compact: minimal spacing (1.05x instead of 1.2x)
        # This is the sweet spot between density and readability
        line_height = int(line_height * 1.2)
    except:
        # Fallback to estimates with compact spacing
        avg_char_width = font_size * 0.6  # Slightly more aggressive
        line_height = int(font_size * 1.2)
    
    result = (avg_char_width, line_height)
    _FONT_METRICS_CACHE[cache_key] = result
    return result


def find_fast_dimensions(
    text: str,
    font,
    font_size: int,
    padding: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    use_precise: bool = False
) -> Tuple[int, int, List[Tuple[str, bool]]]:
    """
    Fast dimension calculation - uses fixed width and calculates required height.
    Significantly faster than binary search approach, suitable for real-time use.
    
    Returns:
        (width, height, wrapped_lines) where wrapped_lines is List[Tuple[str, bool]]
    """
    # Use max_width for consistent, fast layout
    width = max_width
    available_width = width - 2 * padding
    
    # Get accurate font metrics
    avg_char_width, line_height = get_font_metrics(font, font_size)
    
    # Wrap text
    if use_precise:
        lines = wrap_text_precise(text, available_width, font, font_size)
    else:
        max_chars_per_line = int(available_width / avg_char_width)
        lines = wrap_text_fast(text, max_chars_per_line)
    
    # Calculate required height
    num_paragraph_breaks = sum(1 for _, is_para_end in lines if is_para_end)
    required_height = len(lines) * line_height + num_paragraph_breaks * int(line_height * 0.0) + 2 * padding
    
    # Clamp to min/max bounds
    height = max(min_height, min(max_height, required_height))
    
    # Truncate lines if needed
    if required_height > max_height:
        available_height = height - 2 * padding
        max_lines = int(available_height / line_height)
        lines = lines[:max_lines]
    
    return (width, height, lines)


def find_optimal_dimensions(
    text: str,
    font,
    font_size: int,
    padding: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    use_precise: bool = False
) -> Tuple[int, int, List[Tuple[str, bool]]]:
    """
    Find optimal image dimensions using binary search and precise font metrics.
    Goal: Maximum text coverage with minimum resolution while maintaining clarity.
    
    Returns:
        (width, height, wrapped_lines) where wrapped_lines is List[Tuple[str, bool]]
    """
    # Get accurate font metrics
    avg_char_width, line_height = get_font_metrics(font, font_size)
    
    # Calculate text length to estimate optimal starting point
    text_length = len(text.replace('\n', ' '))
    total_text_area = text_length * avg_char_width * line_height
    
    # Start with a square-ish aspect ratio for better packing
    aspect_ratio = 1.5  # Slightly wider than tall for better readability
    estimated_width = int((total_text_area * aspect_ratio) ** 0.5)
    estimated_width = max(min_width, min(max_width, estimated_width))
    
    def evaluate_width(width: int) -> Tuple[int, List[Tuple[str, bool]], bool]:
        """
        Evaluate a given width and return (height, lines, fits).
        Returns fits=True if text fits within max_height.
        """
        available_width = width - 2 * padding
        
        if use_precise:
            lines = wrap_text_precise(text, available_width, font, font_size)
        else:
            max_chars_per_line = int(available_width / avg_char_width)
            lines = wrap_text_fast(text, max_chars_per_line)
        
        # Calculate height considering paragraph spacing (minimal spacing for compact layout)
        num_paragraph_breaks = sum(1 for _, is_para_end in lines if is_para_end)
        required_height = len(lines) * line_height + num_paragraph_breaks * int(line_height * 0.0) + 2 * padding
        fits = required_height <= max_height
        
        return required_height, lines, fits
    
    # Binary search for minimum width that fits all text
    left, right = estimated_width, max_width
    best_solution = None
    best_area = float('inf')

    while left <= right:
        mid = (left + right) // 2
        if mid < left:
            mid = left
        if mid > right:
            mid = right
            
        required_height, lines, fits = evaluate_width(mid)
        
        if fits:
            # Text fits! Try to minimize area
            height = required_height
            height = max(min_height, min(max_height, height))
            area = mid * height
            
            if area < best_area:
                best_area = area
                best_solution = (mid, height, lines)
            
            # Try smaller width
            right = mid - 1
        else:
            # Doesn't fit, need wider
            left = mid + 1

    # If no solution found in binary search, use max dimensions with truncation
    if best_solution is None:
        width = max_width
        height = max_height
        available_width = width - 2 * padding
        
        if use_precise:
            lines = wrap_text_precise(text, available_width, font, font_size)
        else:
            max_chars_per_line = int(available_width / avg_char_width)
            lines = wrap_text_fast(text, max_chars_per_line)
        
        # Truncate lines that don't fit (considering paragraph spacing)
        available_height = height - 2 * padding
        max_lines = int(available_height / line_height)
        lines = lines[:max_lines]
        
        best_solution = (width, height, lines)

    # Final optimization: try to reduce height if there's too much empty space
    width, height, lines = best_solution
    num_paragraph_breaks = sum(1 for _, is_para_end in lines if is_para_end)
    actual_height_needed = len(lines) * line_height + num_paragraph_breaks * int(line_height * 0.0) + 2 * padding
    actual_height_needed = max(min_height, actual_height_needed)

    if actual_height_needed < height:
        height = actual_height_needed

    return (width, height, lines)


def text_to_adaptive_image(
    text: str,
    font_size: int = 8,
    padding: int = 8,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None,
    min_width: int = 28,
    max_width: int = 1024,
    min_height: int = 28,
    max_height: int = 1024,
    use_precise: bool = True,
    fast_mode: bool = True,
    compact_mode: bool = False,
    compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
    highlight_configs: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Image.Image:
    """
    Convert text to image with ultimate optimization for maximum text coverage
    and minimum resolution while maintaining clarity.
    
    Args:
        text: Input text to render
        font_size: Font size (default 8 for dense packing)
        padding: Padding in pixels (optimized for minimal waste)
        bg_color: Background color RGB tuple
        text_color: Text color RGB tuple
        font_path: Custom font path (condensed fonts recommended)
        min_width: Minimum image width
        max_width: Maximum image width
        min_height: Minimum image height
        max_height: Maximum image height
        use_precise: Use precise font measurements (recommended, slightly slower but optimal)
        fast_mode: Use fast mode (fixed width) instead of binary search (much faster)
        compact_mode: Enable compact mode (replace newlines with symbols)
        compact_symbol: Symbol to use for newline replacement in compact mode
        highlight_configs: List of dicts specifying text contexts to highlight with colors.
                          To highlight compact_symbol, include it in highlight_configs.
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "⏎", "color": [128, 128, 128]}]
    
    Returns:
        PIL Image with optimally packed text
    """
    text = text.strip() if text else ""
    
    # Apply compact mode transformation if enabled
    if compact_mode:
        text = apply_compact_mode(text, compact_symbol)
    
    optimized_padding = padding

    min_width = max(min_width, 28)
    max_width = min(max_width, 1024)

    # Use cached font for significant speedup
    font = _get_cached_font(font_path, font_size)

    # Find dimensions - use fast mode for real-time performance
    if fast_mode:
        img_width, img_height, lines = find_fast_dimensions(
            text, font, font_size, optimized_padding, 
            min_width, max_width, min_height, max_height,
            use_precise=use_precise
        )
    else:
        img_width, img_height, lines = find_optimal_dimensions(
            text, font, font_size, optimized_padding, 
            min_width, max_width, min_height, max_height,
            use_precise=use_precise
        )
    
    # Get actual line height from font metrics
    _, line_height = get_font_metrics(font, font_size)

    # Create image with optimized dimensions
    img = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Render text with optimal spacing and 0.5x line spacing after paragraphs
    y_position = optimized_padding
    paragraph_spacing = int(line_height * 0.0)
    
    for line_text, is_paragraph_end in lines:
        if highlight_configs:
            # Render with highlighted contexts (also handles compact_symbol if defined in highlight_configs)
            _render_line_with_highlights(
                draw, line_text, optimized_padding, y_position,
                text_color, highlight_configs, font
            )
        else:
            draw.text((optimized_padding, y_position), line_text, fill=text_color, font=font)
        y_position += line_height
        # Add extra spacing after paragraph end
        if is_paragraph_end:
            y_position += paragraph_spacing
    
    return img


def _get_text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    """
    Get the actual rendering width of text using the most accurate method available.
    
    Args:
        font: Font to use
        text: Text to measure
    
    Returns:
        Width in pixels
    """
    if not text:
        return 0
    try:
        # getlength() is the most accurate method for text width (includes kerning)
        return int(font.getlength(text))
    except AttributeError:
        # Fallback for older PIL versions
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        except:
            return len(text) * 6  # Last resort fallback


def _render_line_with_highlights(
    draw: ImageDraw.ImageDraw,
    line_text: str,
    x: int,
    y: int,
    text_color: Tuple[int, int, int],
    highlight_configs: Optional[List[Dict[str, Any]]],
    font: ImageFont.FreeTypeFont
) -> None:
    """
    Render a line of text with multiple highlighted contexts in different colors.
    
    Uses cumulative prefix width calculation to ensure proper character spacing
    and kerning across segment boundaries.
    
    Args:
        draw: PIL ImageDraw object
        line_text: Text to render
        x: X position
        y: Y position
        text_color: Default color for regular text
        highlight_configs: List of dicts with 'context' and 'color' keys
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "Observation", "color": [0, 255, 0]}]
        font: Font to use
    """
    if not highlight_configs:
        # No highlights, render normally
        draw.text((x, y), line_text, fill=text_color, font=font)
        return
    
    # Build a list of (start_pos, end_pos, color) for all matches
    highlights = []
    for config in highlight_configs:
        context = config.get('context', '')
        color = tuple(config.get('color', text_color))
        if not context:
            continue
        
        # Find all occurrences of this context in the line
        start = 0
        while True:
            pos = line_text.find(context, start)
            if pos == -1:
                break
            highlights.append((pos, pos + len(context), color))
            start = pos + 1
    
    # Sort highlights by start position
    highlights.sort(key=lambda h: h[0])
    
    # Merge overlapping highlights (take the first one in case of overlap)
    merged_highlights = []
    for start, end, color in highlights:
        if merged_highlights and start < merged_highlights[-1][1]:
            # Overlapping, keep the existing one
            continue
        merged_highlights.append((start, end, color))
    
    # If no matches found, render normally
    if not merged_highlights:
        draw.text((x, y), line_text, fill=text_color, font=font)
        return
    
    # Build segments: list of (text, color, start_char_pos, end_char_pos)
    segments = []
    current_pos = 0
    
    for start, end, color in merged_highlights:
        # Add non-highlighted segment before this highlight
        if current_pos < start:
            segments.append((line_text[current_pos:start], text_color, current_pos, start))
        # Add highlighted segment
        segments.append((line_text[start:end], color, start, end))
        current_pos = end
    
    # Add remaining non-highlighted segment
    if current_pos < len(line_text):
        segments.append((line_text[current_pos:], text_color, current_pos, len(line_text)))
    
    # Render each segment using cumulative prefix width for positioning
    # This ensures proper kerning is considered
    for segment_text, segment_color, start_char_pos, end_char_pos in segments:
        # Calculate x position using prefix width (considers kerning)
        prefix = line_text[:start_char_pos]
        segment_x = x + _get_text_width(font, prefix)
        draw.text((segment_x, y), segment_text, fill=segment_color, font=font)


def text_to_adaptive_image_compact(
    text: str,
    font_size: int = 8,
    padding: int = 8,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None,
    min_width: int = 28,
    max_width: int = 1024,
    min_height: int = 28,
    max_height: int = 1024,
    use_precise: bool = False,
    compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
    highlight_configs: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Tuple[Image.Image, int, str, List[Tuple[str, bool]]]:
    """
    Convert text to image in compact mode with caching metadata.
    
    This function is designed for incremental caching in compact mode:
    - Returns the number of complete lines and their character count
    - Returns the incomplete line's text for prepending to next render
    
    Args:
        text: Input text to render (newlines will be replaced with symbols)
        font_size: Font size
        padding: Padding in pixels
        bg_color: Background color RGB tuple
        text_color: Text color RGB tuple
        font_path: Custom font path
        min_width: Minimum image width
        max_width: Maximum image width
        min_height: Minimum image height
        max_height: Maximum image height
        use_precise: Use precise font measurements
        compact_symbol: Symbol to use for newline replacement
        highlight_configs: List of dicts specifying text contexts to highlight with colors.
                          To highlight compact_symbol, include it in highlight_configs.
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "⏎", "color": [128, 128, 128]}]
    
    Returns:
        Tuple of:
        - img: PIL Image with rendered text
        - num_complete_lines: Number of lines that filled the available width
        - incomplete_text: Text from the last incomplete line (for next render)
        - lines: List of (line_text, is_complete) tuples
    """
    text = text.strip() if text else ""
    
    # Apply compact mode transformation
    compact_text = apply_compact_mode(text, compact_symbol)
    
    optimized_padding = padding
    min_width = max(min_width, 28)
    max_width = min(max_width, 1024)
    
    # Use cached font
    font = _get_cached_font(font_path, font_size)
    
    # Get font metrics
    avg_char_width, line_height = get_font_metrics(font, font_size)
    
    # Use fixed width (fast mode) for compact rendering
    width = max_width
    available_width = width - 2 * padding
    max_chars_per_line = int(available_width / avg_char_width)
    
    # Wrap text with compact mode logic
    lines, complete_char_count, incomplete_text = wrap_text_compact(
        compact_text, max_chars_per_line
    )
    
    # Count complete lines
    num_complete_lines = sum(1 for _, is_complete in lines if is_complete)
    
    # Calculate required height
    required_height = len(lines) * line_height + 2 * padding
    height = max(min_height, min(max_height, required_height))
    
    # Truncate lines if needed
    if required_height > max_height:
        available_height = height - 2 * padding
        max_lines = int(available_height / line_height)
        lines = lines[:max_lines]
        # Recalculate complete lines and incomplete text
        num_complete_lines = sum(1 for _, is_complete in lines if is_complete)
        if lines:
            _, last_is_complete = lines[-1]
            if not last_is_complete:
                incomplete_text = lines[-1][0]
    
    # Create image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Render text with colored symbols and highlights
    y_position = optimized_padding
    
    for line_text, _ in lines:
        if highlight_configs:
            # Render with highlighted contexts (also handles compact_symbol if defined in highlight_configs)
            _render_line_with_highlights(
                draw, line_text, optimized_padding, y_position,
                text_color, highlight_configs, font
            )
        else:
            draw.text((optimized_padding, y_position), line_text, fill=text_color, font=font)
        y_position += line_height
    
    return img, num_complete_lines, incomplete_text, lines


def trajectory_to_image(
    trajectory_text: str,
    font_size: int = 8,
    padding: int = 8,
    use_precise: bool = True,
    fast_mode: bool = True,
    compact_mode: bool = False,
    compact_symbol: str = COMPACT_NEWLINE_SYMBOL,
    highlight_configs: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Image.Image:
    """
    Transform trajectory text to image with ultimate optimization.
    Achieves maximum text coverage with minimum resolution while maintaining clarity.
    
    Args:
        trajectory_text: Trajectory text to render
        font_size: Font size (default 8 for optimal density)
        padding: Padding in pixels (optimized to 8 for minimal waste)
        use_precise: Use precise font measurements for optimal packing (recommended)
        fast_mode: Use fast mode (fixed width) for real-time performance (default True)
        compact_mode: Enable compact mode (replace newlines with symbols)
        compact_symbol: Symbol to use for newline replacement in compact mode
        highlight_configs: List of dicts specifying text contexts to highlight with colors.
                          To highlight compact_symbol, include it in highlight_configs.
                          Example: [{"context": "Action", "color": [255, 0, 0]}, 
                                   {"context": "⏎", "color": [128, 128, 128]}]
        **kwargs: Additional parameters passed to text_to_adaptive_image

    Returns:
        PIL Image object with optimally packed text
    """
    trajectory_text = trajectory_text.strip() if trajectory_text else ""
    formatted_text = trajectory_text

    return text_to_adaptive_image(
        formatted_text,
        font_size=font_size,
        padding=padding,
        use_precise=use_precise,
        fast_mode=fast_mode,
        compact_mode=compact_mode,
        compact_symbol=compact_symbol,
        highlight_configs=highlight_configs,
        **kwargs
    )