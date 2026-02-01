"""
Environment configuration for ML libraries.

This module MUST be imported before any ML libraries (sentence_transformers,
transformers, huggingface_hub) because they read environment variables at
import time.

Usage:
    import config  # Always import first
    from sentence_transformers import SentenceTransformer
"""

import logging
import os
import warnings

# =============================================================================
# HUGGING FACE HUB CONFIGURATION
# =============================================================================

# Disable download progress bars for cleaner output
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Disable anonymous usage telemetry to Hugging Face
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Don't try to auto-detect HF tokens from the system
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# Only show errors from Hugging Face Hub, not info/warnings
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Disable symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# =============================================================================
# TRANSFORMERS CONFIGURATION
# =============================================================================

# Only show errors from transformers library
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Disable tokenizer parallelism warning (not needed for single-threaded use)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =============================================================================
# MLX CONFIGURATION (Apple Silicon)
# =============================================================================

# On Apple Silicon (M1/M2/M3), sentence-transformers may automatically use
# Apple's MLX framework as a backend instead of PyTorch for faster inference.
# We don't use MLX directly, but it runs behind the scenes.
# MLX lazy loading prints verbose "LOAD REPORT" messages during inference.
# Setting to "0" loads weights upfront during init, suppressing this output.
# This setting has no effect on Windows/Linux or Intel Macs.
os.environ.setdefault("MLX_LAZY_LOAD", "0")

# =============================================================================
# PYTHON WARNINGS FILTER
# =============================================================================

# Filter out specific warning messages by their text content
warnings.filterwarnings("ignore", message=".*position_ids.*")  # MLX model loading artifact
warnings.filterwarnings("ignore", message=".*unauthenticated.*")  # HF token reminder
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")  # HF token reminder variant
warnings.filterwarnings("ignore", category=FutureWarning)  # Deprecation notices
warnings.filterwarnings("ignore", category=UserWarning)  # General user warnings

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Set logging level to ERROR for noisy libraries (hides INFO and WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# =============================================================================
# OPTIONAL: HUGGING FACE AUTHENTICATION
# =============================================================================
# Set the HF_TOKEN environment variable to authenticate with Hugging Face Hub
# for higher rate limits and faster downloads.
# Get your token at: https://huggingface.co/settings/tokens
#
# Example (bash):  export HF_TOKEN=your_token_here
# Example (Windows): set HF_TOKEN=your_token_here
