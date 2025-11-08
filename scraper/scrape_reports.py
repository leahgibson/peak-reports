"""
Scraper for 14ers.com trip reports/peak status posts.
Collectes recent condition reports for al 53 CO 14ers.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportScraper:
    """Scrapes trip reports from 14ers.com"""

    BASE_URL = "https://www.14ers.com/php14ers/peakstatus_peak.php?peakparm=100{:02d}"
    OUTPUT_DIR = Path("scraper/data/reports")
    DELAY_BETWEEN_REQUESTS = 2 

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def scrape_all_peaks(self, start_peak=1, end_peak=53):
        """
        Scrape reports for all peaks.
        
        Parameters:
        - start_peak: Starting peak number
        - end_peak: Ending peak number
        """
        all_reports = []

        for peak_num in range(start_peak, end_peak+1):
            logger.info(f"Scraping peak {peak_num}/{end_peak}...")

            try:
                reports = self.scrape_peak(peak_num)
                all_reports.extend(reports)
                logger.info(f" Found {len(reports)} reports")

                time.sleep(self.DELAY_BETWEEN_REQUESTS)
            
            except Exception as e:
                logger.error(f"Error scraping peak {peak_num}: {e}")
                continue
        
        logger.info(f"Scraping complete! Saving {len(all_reports)} reports")
        self._save_reports(all_reports)
    
    def scrape_peak(self, peak_num):
        """
        Scrape reports for a given peak
        
        Parameters:
        - peak_num: Peak number (1-53)
        
        Returns:
            List of report dicts
        """
        url = self.BASE_URL.format(peak_num)

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []
    
        report_page = BeautifulSoup(response.content, 'html.parser')
        
        reports = self._parse_reports(report_page, url, peak_num)

        return reports
    
    def _parse_reports(self, soup, url, peak_num):
        """
        Parse trip reports from the page HTML
        
        Parameters:
        - soup: BeautifulSoup object
        - url: Source URL
        - peak_num: Peak number (1-53)
        
        Returns:
            List of report dicts
        """
        reports = []

        peak_name_tag = soup.find('h1')
        peak_name = peak_name_tag.get_text(strip=True) if peak_name_tag else f"Peak_{peak_num:02d}"

        rows = soup.find_all('tr', onclick=lambda x: x and 'recnum' in x)

        for row in rows:
            try:
                td = row.find('td')
                if not td:
                    continue

                # Trip date
                trip_date_div = td.find('div', class_='linkButton')
                trip_date = trip_date_div.get_text(strip=True) if trip_date_div else ""

                # Route
                route = ""
                route_span = td.find('span', class_='bold1', string='Route:')
                if route_span:
                    route_text = route_span.next_sibling
                    if route_text:
                        route = route_text.strip().split('<br/>')[0].strip()
                
                # Posted date
                posted_date = ""
                posted_span = td.find('span', class_='bold1', string='Posted On:')
                if posted_span:
                    posted_text = posted_span.next_sibling
                    if posted_text:
                        posted_date = posted_text.strip().split(',')[0].strip()
                
                # Info
                content = ""
                info_span = td.find('span', class_='bold1', string='Info:')
                if info_span:
                    content_text = info_span.next_sibling
                    if content_text:
                        content = content_text.strip()
                
                if trip_date and content:
                    report = {
                        "peak_name": peak_name,
                        "route": route,
                        "date_posted": posted_date,
                        "trip_date": trip_date,
                        "content": content,
                        "url": url,
                        "peak_number": peak_num,
                    }
                    reports.append(report)
            
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue

        return reports
    
    def _save_reports(self, reports):
        """Save reports to JSON file"""
        output_file = self.OUTPUT_DIR / f"reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(reports)} reports to {output_file}")

def main():
    scraper = ReportScraper()

    scraper.scrape_all_peaks(start_peak=1, end_peak=53)

if __name__ == "__main__":
    main()





