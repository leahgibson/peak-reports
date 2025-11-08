"""
14er trail report NLP pipeline
Combined rule-based extraction with zero-shot classification
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from transformers import pipeline

class TrailReportProcessor:
    """Process 14er trail reports to extract conditions and status"""

    def __init__(self):

        print("Loading ero-shot classification model...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-nmli",
            device=-1
        )
        print("Model loaded sucessfully!")

        # Define etraction patterns
        self.equipment_kw = {}

        self.trail_condition_kw = {}

        self.difficulty_kw = {}

        self.warning_kw = {}

    