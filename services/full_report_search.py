import re
from typing import Optional
from services.full_report_store import load_full_report_text

def search_full_report(report_id: str, query: str) -> Optional[str]:
    """
    Search for a query string in the full report text and return the first matching line or context.
    """
    text = load_full_report_text(report_id)
    if not text:
        return None
    # Simple case-insensitive search for the query
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for line in text.splitlines():
        if pattern.search(line):
            return line.strip()
    return None
