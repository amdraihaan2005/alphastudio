import os
import json
import re

# Mapping of common NSE Ticker symbols to Full Company Names
TICKER_MAP = {
    "RELIANCE": "Reliance Industries Limited",
    "TCS": "Tata Consultancy Services Limited",
    "INFY": "Infosys Limited",
    "HDFCBANK": "HDFC Bank Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "BHARTIARTL": "Bharti Airtel Limited",
    "SBIN": "State Bank of India",
    "LICI": "Life Insurance Corporation of India",
    "ITC": "ITC Limited",
    "HINDUNILVR": "Hindustan Unilever Limited"
}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    downloads_dir = os.path.join(base_dir, "data", "downloads")
    
    # Create the folder if it doesn't exist yet
    os.makedirs(downloads_dir, exist_ok=True)
    
    print(f"Scanning directory: {downloads_dir} for PDF files...")
    
    files = [f for f in os.listdir(downloads_dir) if f.endswith(".pdf")]
    
    if not files:
        print("\n[!] No PDF files found in data/downloads/ yet.")
        print("Please download annual report PDFs from the NSE website and save them in:")
        print(f"--> {downloads_dir}")
        print("\nName your files like: TICKER_YEAR.pdf (e.g., RELIANCE_2024.pdf or TCS_2024.pdf)")
        return

    manifest = []
    
    for filename in files:
        # Expected pattern: TICKER_YEAR.pdf (e.g. RELIANCE_2024.pdf)
        match = re.match(r"^([A-Z0-9]+)_([0-9]{4})\.pdf$", filename, re.IGNORECASE)
        
        if match:
            ticker = match.group(1).upper()
            year = int(match.group(2))
            company_name = TICKER_MAP.get(ticker, f"{ticker} Limited")
            
            manifest.append({
                "ticker": ticker,
                "company": company_name,
                "year": year,
                "type": "Annual Report",
                "filename": filename
            })
            print(f"Added to manifest: {filename} -> {company_name} ({year})")
        else:
            print(f"[Warning] Filename '{filename}' does not match pattern TICKER_YEAR.pdf. Skipping.")

    # Write manifest.json
    manifest_path = os.path.join(downloads_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\n[+] Created manifest.json at {manifest_path} with {len(manifest)} entries.")

if __name__ == "__main__":
    main()
