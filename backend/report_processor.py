"""
14er trail report NLP pipeline
Combined rule-based extraction with zero-shot classification
"""

import re
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import json
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

class TrailReportProcessor:
    """Process 14er trail reports to extract conditions and status"""

    def __init__(self, similarity_threshold=0.75):

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

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB, very fast
        self.similarity_threshold = similarity_threshold

        print("Model loaded sucessfully!")
    
    def calculate_temporal_weight(self, trip_date_str: str, reference_date: datetime = None):
        """Calculate temperal weight of report based on report age"""

        if reference_date is None:
            reference_date = datetime.now()
        
        try:
            trip_date = datetime.strptime(trip_date_str, "%m/%d/%Y")
        except Exception:
            return 0.0 # no weight for unparsable date
        
        days_ago = (reference_date - trip_date).days

        if days_ago <= 3:
            return 1.0
        elif days_ago <= 7:
            return 0.8
        elif days_ago <= 14:
            return 0.5
        elif days_ago <= 30:
            return 0.2
        elif days_ago <= 45:
            return 0.0
        else:
            return 0.0
    
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
        elevation_prompt = f"""Read this hiking trip report and identify at what elevation the conditions changed or became notable (e.g., "snow above 13000 ft", "at treeline"). Return only the condition and the associated elevation description or "not specified". Trip report: {content}"""
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

    def group_terms(self, terms_dict: Dict):
        """
        Groups processed report using semantic similarity to minimize similar terms (i.e. high wings & windy)

        Returns: 
        dict where key is canonical term and value is list of weights from similar terms
        """

        terms = list(terms_dict.keys())
        weights = list(terms_dict.values())

        if len(terms) == 1:
            return terms_dict
        
        # Encode all terms into embeddings
        embeddings = self.embedder.encode(terms)

        # Compute pairwise cosine similarities
        n = len(terms)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Group similar terms
        groups = []
        assigned = set()

        for i in range(n):
            if i in assigned:
                continue

            group = [i]
            assigned.add(i)

            # Find all terms similar to (or transitively similar) to term i
            queue = [i]
            while queue:
                current = queue.pop(0)

                for j in range(n):
                    if j not in assigned and similarity_matrix[current, j] >= self.similarity_threshold:
                        group.append(j)
                        assigned.add(j)
                        queue.append(j)
            
            groups.append(group)
        
        print(groups)
        
        if len(assigned) < n:
            print("WARNING: Not all items assigned!")
            print(groups)
        
        # For each group, assign canonical name and list of weights
        new_terms_dict = {}
        for group in groups:
            # Canonical name goes to whichever embedding is most similar to all other items
            # Computed by selecting max(sum(embeddings))
            # If tie, select first entry
            group_matrix = similarity_matrix[np.ix_(group, group)]
            col_sums = group_matrix.sum(axis=0)
            max_index = np.argmax(col_sums)
            canonical_name = terms[group[max_index]] 

            group_weights = [weights[i] for i in group]

            new_terms_dict[canonical_name] = group_weights
        
        return new_terms_dict


    def aggregate_route_reports(self, reports: List[Dict], peak_name, route):
        """
        Aggregate multiple reports (for same peak) for a singel route with temporal weighting

        All reports should be for the same peak & route
        """

        if not reports:
            return None
        
        print(f"Aggregating {len(reports)} reports for {peak_name} - {route}")

        # Calculate weights based on report date
        weighted_reports = []
        equipment_dict = defaultdict(list)
        trail_conditions_dict = defaultdict(list)
        warnings_dict = defaultdict(list)

        elevation_report = []

        for report in reports:
            weight = self.calculate_temporal_weight(report['trip_date'])
            weighted_reports.append({
                'report': report,
                'weights': weight
            })

            equipment_dict[report["equipment"]].append(weight)
            trail_conditions_dict[report["trail_conditions"]].append(weight)
            warnings_dict[report["warnings"]].append(weight)

            # Add elevation context if most recent week
            if weight >= 0.7:
                elevation_report.append(report["elevation_context"])
        
        # Identify canonical terms
        equipment_groups = self.group_terms(equipment_dict)
        trail_conditions_groups = self.group_terms(trail_conditions_dict)
        warnings_groups = self.group_terms(warnings_dict)

        # Generate equipment list by recency
        equipment = []
        for item, weights in equipment_groups.items():
            if max(weights) >= 0.2:
                equipment.append(item)
        
        # Trail conditions determined by most reported & most recent
        weighted_conditions = {key: sum(values) for key, values in trail_conditions_groups.items()}
        conditions = list(weighted_conditions.keys())
        condition_weights = list(weighted_conditions.values())
        max_weight = max(condition_weights)
        max_indices = [i for i, x in enumerate(condition_weights) if x == max_weight]
        trail_conditions = [conditions[i] for i in max_indices]

        # Warnings come from most recent week of reports
        warnings = []
        for warning, weights in warnings_groups.items():
            if max(weights) >= 0.7:
                warnings.append(warning)
        
        # Build full result
        aggregated_report = {
            'peak_name': peak_name,
            'route': route,
            'num_reports': len(reports),
            'date_range': {
                'oldest': min(r['trip_date'] for r in reports),
                'newest': max(r['trip_date'] for r in reports)
            },
            'trail_conditions': trail_conditions,
            'equipment': equipment,
            'warnings': warnings,
            'elevation_context': elevation_report
        }

        return aggregated_report
    
    def process_reports(self):
        """Processes all reports.
        
        TODO:
        - group by route
        - pass to aggregator
        """






        


        



        
        

    


