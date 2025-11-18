"""
14er trail report NLP pipeline
Combined rule-based extraction with zero-shot classification
"""

import re
import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class TrailReportProcessor:
    """Process 14er trail reports to extract conditions and status"""

    def __init__(self, similarity_threshold=0.75):

        load_dotenv()

        print("Logging in...")
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
        else:
            print("No HuggingFace Token found.")

        print("Loading model...")
        model_name = "meta-llama/Llama-3.2-3B-Instruct" 
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            max_new_tokens=512
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
        equipment_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Read this hiking trip report and list ONLY the equipment that was mentioned. Return as a comma-separated list with no other text. Trip report: {content}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        equipment_response = self.generator(equipment_prompt)[0]['generated_text']
        equipment_response = equipment_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        equipment_response = equipment_response.replace("<|eot_id|>", "").strip()

        print(equipment_response)
        exit()
        equipment_list = [item.strip() for item in equipment_response.split(',') if item.strip()]
        
        # Extract trail conditions
        conditions_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Read this hiking trip report and describe the trail surface conditions in 3-5 words (for example, snow-covered, icy, muddy, etc.). Return only the condition description. Trip report: {content}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        conditions_response = self.generator(conditions_prompt, num_return_sequences=1)[0]['generated_text']
        trail_conditions = conditions_response.strip()

        # Extract warnings
        warnings_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Read this hiking trip report and list any hazards, warnings, or dangerous conditions mentioned (e.g., high wind gusts, avalanche risk, freezing). Return as a comma-separated list or "none" if no warnings. Trip report: {content}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        warnings_response = self.generator(warnings_prompt, num_return_sequences=1)[0]['generated_text']
        if warnings_response.strip().lower() == "none":
            warnings_list = []
        else:
            warnings_list = [item.strip() for item in warnings_response.split(',') if item.strip()]
        

        # Extract elevation info
        elevation_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Read this hiking trip report and identify at what elevation the conditions changed or became notable (e.g., "snow above 13000 ft", "at treeline"). Return only the condition and the associated elevation description or "none". Trip report: {content}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        elevation_response = self.generator(elevation_prompt, num_return_sequences=1)[0]['generated_text']
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

        print(extraction)
        exit()

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
            'elevation_context': elevation_report,
            'update_date': datetime.today().strftime("%m/%d/%Y")
        }

        return aggregated_report
    
    def process_all_reports(self, reports: List[Dict]):
        """Processes all reports."""

        print(f"\n{'='*60}")
        print(f"Processing {len(reports)} total reports")
        print(f"{'='*60}")

        # Extract info from each report
        processed_reports = []
        for i, report in enumerate(reports):
            print(f"[{i+1}/{len(reports)}]")
            try:
                processed = self.process_report(report)
                processed_reports.append(processed)
            except Exception as e:
                print(f" ERROR: {str(e)}")
                continue
        
        # Group reports by peak
        by_peak = defaultdict(list)
        for report in processed_reports:
            by_peak[report['peak_name']].append(report)
        
        # For each peak, group by route
        peak_route_groupings = {}
        for peak_name, reports in by_peak.items():
            peak_route_groupings[peak_name] = defaultdict(list)
            for report in reports:
                peak_route_groupings[peak_name][report["route"]].append(report)
        
        # Aggregate route reports
        aggregated_results = []
        for peak_name, route_reports in peak_route_groupings.items():
            for route, reports in route_reports.items():
                aggregated_results.append(self.aggregate_route_reports(self, reports, peak_name, route))
        

        return aggregated_results
    

if __name__ == "__main__":

    print("Loading reports...")
    # Load json of reports
    with open("scraper/data/reports/reports_20251107_143336.json", 'r') as file:
        reports = json.load(file)
    print("Reports loaded")

    processor = TrailReportProcessor()

    results = processor.process_all_reports(reports)

    # Save results
    output_file = "route_status_results.json"
    output_path = os.path.join("backend/data", output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved.")





### NOTE: for future implmentation, will save past terms and weights and just pass this to update status###











        


        



        
        

    


