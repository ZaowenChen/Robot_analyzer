"""
ROSBag Diagnostic Extractor

Compresses large Gaussian robot ROS logs into LLM-ready diagnostic signal
by treating logs as polyrhythmic patterns and extracting only deviations.

Supports both text log files and binary .bag files.
"""

__version__ = "0.1.0"
