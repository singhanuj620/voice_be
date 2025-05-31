import os
from typing import Optional

REPORTS_DIR = "reports_fulltext"

# Ensure the directory exists
def ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)

def save_full_report_text(report_id: str, text: str):
    """
    Save the full text of a report using a unique report_id.
    """
    ensure_reports_dir()
    file_path = os.path.join(REPORTS_DIR, f"{report_id}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def load_full_report_text(report_id: str) -> Optional[str]:
    """
    Retrieve the full text of a report by report_id.
    """
    file_path = os.path.join(REPORTS_DIR, f"{report_id}.txt")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
