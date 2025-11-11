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
    
    

