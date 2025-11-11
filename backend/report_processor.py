"""
14er trail report NLP pipeline
Combined rule-based extraction with zero-shot classification
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class TrailReportProcessor:
    """Process 14er trail reports to extract conditions and status"""

    def __init__(self):

        print("Loading models...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-nmli",
            device=-1
        )

        model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text2text-generation",
            model=self.generator_model,
            tokenizer=self.tokenizer,
            device=-1,
            max_lenth=512
        )

        print("Model loaded sucessfully!")
    
    def calculate_temporal_weight(self, trip_date_str: str, reference_date: datetime = None):
        """Calculate temperal weight of report based on report age"""

        if reference_date is None:
            reference_date = datetime.now()
        
        try:
            trip_date = datetime.strptime(trip_date_str, "%m/%d/%Y")
        except Exception:
            return 0.1 # very low weight for unparsable date
        
        days_ago = (reference_date - trip_date).days

        if days_ago <= 3:
            return 1.0
        elif days_ago <= 7:
            return 0.7
        elif days_ago <= 14:
            return 0.4
        elif days_ago <= 30:
            return 0.2
        else:
            return 0.1
        
    


