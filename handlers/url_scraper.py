import re
import requests
from bs4 import BeautifulSoup

def extract_url(text: str) -> str:
    """Extract the first URL from the provided text."""
    url_match = re.search(r'https?://\S+', text)
    return url_match.group(0) if url_match else None

def scrape_url(url: str) -> str:
    """Scrape and return visible text and all tables from a web page."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract visible text content
        for script in soup(["script", "style"]):
            script.decompose()
        text = ' '.join(soup.stripped_strings)[:3000]

        # Log available tables for debugging
        tables = []
        try:
            import pandas as pd
            tables = pd.read_html(response.text)
            if tables:
                table_summary = f"\n\nExtracted {len(tables)} HTML tables from the URL."
                text += table_summary
        except Exception as e:
            text += f"\n\n[pandas.read_html() failed: {e}]"

        return text
    except Exception as e:
        return f"[Error scraping URL: {e}]"
