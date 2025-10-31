"""
14ers.com Static Trail Info Scraper
Collects basic trail data for all Colorado 14ers
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
from typing import List, Dict
import re
import os

class FourteenerScraper:
    """Scraper for static trail information from 14ers.com"""

    BASE_URL = "https://www.14ers.com"
    PEAKS_URL_TEMPLATE = f"{BASE_URL}/peaks/100{{:02d}}"
    PEAKS_URL = f"{BASE_URL}/14ers"
    ROUTES_URL = f"{BASE_URL}/routes.php"

    def __init__(self, delay: float = 2.0):
        """
        Initialize scraper
        
        Parameters:
        - delay: seconds to wait between requests
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'peak-reports-scraper/1.0 (personal project for trail recommendations)'})

    
    def get_fourteener_urls(self) -> List[str]:
        """
        Scrape the main routes page to get list of all 14ers with their URLs

        Returns:
            List of URLs for peaks
        """

        urls = [self.PEAKS_URL_TEMPLATE.format(i) for i in range (1, 54)]
        print(f"Generated {len(urls)} peak URLs")
        return urls
    
    def scrape_trail_details(self, peak_url: str) -> Dict:
        """
        Scrape detailed trail information for a specific 14ers

        Parameters:
        - peak_url: URL to the peak's page

        Returns:
            Dict with trail details
        """
        peak_num = peak_url.split('/')[-1]
        print(f"Scraping peak {peak_num}...")
        time.sleep(self.delay)

        try:
            response = self.session.get(peak_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching peak {peak_num}: {e}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        peak_name = None
        h1 = soup.find('h1')
        if h1:
            header = h1.get_text(strip=True)
            peak_name = header.split('-')[0].strip()
        
        # Initialize trail data
        trail_data = {
            'name': peak_name,
            'url': peak_url,
            'elevation': None,
            'rank': None,
            'range': None,
            'lat': None,
            'lon': None,
            'county': None,
            'routes': [],
        }

        # Scrape peak info
        sidebar = soup.find('div', id='sidebar')

        if sidebar:
            sidebar_text = sidebar.get_text()

            # Elevation
            elev_match = re.search(r'Elevation(\d{2},\d{3}).*?\(LiDAR\)', sidebar_text)
            if not elev_match:
                elev_match = re.search(r'(\d{2},\d{3})', sidebar_text)
            if elev_match:
                trail_data['elevation'] = elev_match.group(1)

            # Rank
            rank_match = re.search(r'CO 14er Rank(\d+)\s+of\s+\d+', sidebar_text)
            if rank_match:
                trail_data['rank'] = int(rank_match.group(1))
            
            # Mountain Range
            range_match = re.search(r'Range([A-Za-z\s]+?)(?:Forest|Lat/Lon|$)', sidebar_text)
            if range_match:
                trail_data['range'] = range_match.group(1).strip()
            
            # Lat/Lon
            latlon_match = re.search(r'Lat/Lon([-\d.]+),\s*([-\d.]+)', sidebar_text)
            if latlon_match:
                trail_data['lat'] = float(latlon_match.group(1))
                trail_data['lon'] = float(latlon_match.group(2))
            
            # County
            county_match = re.search(r'County([A-Za-z\s]+?)(?:Towns|$)', sidebar_text)
            if county_match:
                trail_data['county'] = county_match.group(1).strip()

        else:
            print("Warning: Could not find sidebar.")
        
        # Scrape routes info
        routes_url = peak_url + "?t=routes"
        print(f"Fetching routes from {routes_url}...")
        time.sleep(self.delay)

        try:
            routes_response = self.session.get(routes_url, timeout=10)
            routes_response.raise_for_status()
            routes_soup = BeautifulSoup(routes_response.content, 'html.parser')

            routes_content = routes_soup.find('div', id='tab2-content')

            if routes_content:
                trail_data['routes'] = self.extract_routes(peak_num, peak_url)
            else:
                print("Warning: Could not find tab2-content div.")

        except requests.RequestException as e:
            print(f"Error fetching routes: {e}")
        
        return trail_data
    
    def extract_routes(self, peak_id, peak_url) -> List[Dict]:
        """
        Extract individual route information from the routes tab content
        
        Parameters:
            peak_id: peak number identifier 100xx
            peak_url: url to peak page
            
        Returns:
            List of route dictionaries
        """
        
        url = "https://www.14ers.com/php14ers/ajax_peak_getroutes.php"
        try:
            params = {"peakid": peak_id}

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': peak_url,
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': '*/*'
            }
            time.sleep(self.delay)

            r = requests.get(url, params=params, headers=headers, timeout=10)
            routes_output = BeautifulSoup(r.text, "html.parser")

            routes_table = routes_output.find('table', id='routeResults')
            if not routes_table:
                print(f"Warning: No routes table found for peak {peak_id}")
                return []
            
            routes = []

            route_rows = routes_table.find_all('tr')
            for row in route_rows:
                # Description column
                desc_col = row.find('td', class_='colDescription1')
                if not desc_col:
                    continue

                # Route Name
                name_div = desc_col.find('div', class_='linkButton')
                route_name = name_div.get_text(strip=True) if name_div else None
                if not route_name:
                    continue

                # Details
                details_div = desc_col.find('div', style=lambda x:x and 'white-space' in x)
                details_text = details_div.get_text() if details_div else ""

                route_data = {
                    'name': route_name,
                    'class': None,
                    'elevation_gain': None,
                    'distance': None,
                    'is_standard': False,
                    'is_snow_climb': False
                }

                # Class of route
                class_span = desc_col.find('span', class_=re.compile(r'class\d+'))
                if class_span:
                    class_text = class_span.get_text(strip=True)
                    class_match = re.search(r'Class\s+(\d+)', class_text)
                    if class_match:
                        route_data['class'] = int(class_match.group(1))
                
                # Elevation gain
                gain_match = re.search(r'Total Elevation Gain:\s*([\d,]+)', details_text)
                if gain_match:
                    route_data['elevation_gain'] = int(gain_match.group(1).replace(',', ''))
                
                # Distance
                dist_match = re.search(r'Round-trip Distance:\s*([\d.]+)\s*mi', details_text)
                if dist_match:
                    route_data['distance'] = float(dist_match.group(1))
                
                # Check for standard route & snow climbs (star and snowflake)
                icons_col = row.find('td', class_='colIcons')
                if icons_col:
                    if icons_col.find('span', class_='fa-star') or icons_col.find('span', class_='fa-solid fa-star'):
                        route_data['is_standard'] = True
                    if icons_col.find('img', title='Snow Climb'):
                        route_data['is_snow_climb'] = True
                
                routes.append(route_data)
            
            print(f"Found {len(routes)} routes")

            return routes
    
        except requests.RequestException as e:
            print(f"Error fetching routes for {peak_id}: {e}")
            return []
        

    def scrape_all_trails(self, output_file: str = 'scraper/data/trails.json'):
        """
        Scrape all 14ers trail info and save to JSON

        Parameters:
        - output_file: Path to save JSON output
        """
        print("Starting 14ers.com trail info scraper...")
        print("=" * 60)

        peak_urls = self.get_fourteener_urls()

        print(f"\nScraping detailed info for {len(peak_urls)} peaks...")
        print("=" * 60)

        all_trails = []

        for i, peak_url in enumerate(peak_urls, 1):
            print(f"\n[{i}/{len(peak_urls)}] {peak_url}")
            trail_data = self.scrape_trail_details(peak_url)

            if trail_data and trail_data['name']:
                all_trails.append(trail_data)
            else:
                print(f"Warning: No data found for {peak_url}")
        
        # Save all data to JSON
        print(f"\n{'=' * 60}")
        print(f"Scraped {len(all_trails)} trails successfully")
        print(f"Saving to {output_file}...")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_trails, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {output_file}")


def main():
    """Run the scraper"""
    scraper = FourteenerScraper(delay=2.0)
    scraper.scrape_all_trails()

if __name__ == "__main__":
    main()



        
            

