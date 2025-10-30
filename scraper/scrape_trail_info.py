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
            'scraped_at': datetime.now
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
                exit()

        except requests.RequestException as e:
            print(f"Error fetching routes: {e}")
        
        exit()
        
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
        params = {"peakid": peak_id}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': peak_url,
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': '*/*'
        }
        time.sleep(self.delay)

        r = requests.get(url, params=params, headers=headers, timeout=10)
        routes_text = BeautifulSoup(r.text, "html.parser")

        print(routes_text)

        ### TO DO: parse this text to get the info I want ###

        exit()



        




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

def main():
    """Run the scraper"""
    scraper = FourteenerScraper(delay=2.0)
    scraper.scrape_all_trails()

if __name__ == "__main__":
    main()



        
            

