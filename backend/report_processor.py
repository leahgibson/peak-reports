"""
14er trail report NLP pipeline
Combined rule-based extraction with zero-shot classification
"""

import re
from datetime import datetime
from typing import List, Dict, Any
import json

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class TrailReportProcessor:
    """Process 14er trail reports to extract conditions and status"""

    def __init__(self):

        print("Loading model...")
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
    
    def process_report(self, report):
        """Process and extract structured information via prompts"""
        print(f"\nProcessing: {report['peak_name']} - {report['route']} ({report['trip_date']})")

        content = report["content"]

        # Extract equipment
        equipment_prompt = f"""Read this hiking trip report and list ONLY the equipment that was mentioned as needed, required, recommended, or used. Return as a comma-separated list with no other text. Trip report: {content}"""
        equipment_response = self.generator(equipment_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        equipment_list = [item.strip() for item in equipment_response.split(',') if item.strip()]
        
        # Extract trail conditions
        conditions_prompt = f"""Read this hiking trip report and describe the trail surface conditions in 3-5 words (e.g., "snow covered icy", "dry and rocky", "muddy wet"). Return only the condition description. Trip report: {content}"""
        conditions_response = self.generator(conditions_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        trail_conditions = conditions_response.strip()

        # Extract warnings
        warnings_prompt = f"""Read this hiking trip report and list any hazards, warnings, or dangerous conditions mentioned (e.g., high winds, avalanche risk, turned back). Return as a comma-separated list or "none" if no warnings. Trip report: {content}"""
        warnings_response = self.generator(warnings_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        if warnings_response.strip().lower() == "none":
            warnings_list = []
        else:
            warnings_list = [item.strip() for item in warnings_response.split(',') if item.strip()]
        

        # Extract elevation info
        elevation_prompt = f"""Read this hiking trip report and identify at what elevation the conditions changed or became notable (e.g., "above 13000 ft", "at treeline"). Return only the elevation description or "not specified". Trip report: {content}"""
        elevation_response = self.generator(elevation_prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        elevation_context = elevation_response.strip()

        extraction = {
            'peak_name': report['peak_name'],
            'route': report['route'],
            'trip_date': report['trip_date'],
            'date_posted': report['date_posted'],
            'equipment': equipment_list,
            'trail_conditions': trail_conditions,
            'warnings': warnings_list,
            'elevation_context': elevation_context,
        }

        return extraction


    


