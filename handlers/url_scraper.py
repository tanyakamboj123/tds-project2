import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

def extract_url(text: str) -> str:
    """Extract the first URL from the provided text."""
    url_match = re.search(r'https?://\S+', text)
    return url_match.group(0) if url_match else None

def scrape_url(url: str, max_chars: int = 3000, max_rows: int = 15) -> str:
    """Scrape and return visible text + normalized table previews from a web page."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract visible text
        for script in soup(["script", "style"]):
            script.decompose()
        text = ' '.join(soup.stripped_strings)[:max_chars]

        # Extract tables
        try:
            tables = pd.read_html(response.text)
            preview_parts = []
            for i, df in enumerate(tables[:3]):  # limit to first 3 tables
                df.columns = [str(c).strip().lower() for c in df.columns]
                preview_parts.append(f"\n\n--- Table {i+1} (first {max_rows} rows) ---\n")
                preview_parts.append(df.head(max_rows).to_csv(index=False))
            if preview_parts:
                text += "\n\n" + "".join(preview_parts)
        except Exception as e:
            text += f"\n\n[pandas.read_html() failed: {e}]"

        return text
    except Exception as e:
        return f"[Error scraping URL: {e}]"
