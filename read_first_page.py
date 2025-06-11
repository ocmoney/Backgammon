import os
from PyPDF2 import PdfReader
import re

def clean_text(text):
    """Clean extracted text by adding spaces between words and removing excessive whitespace"""
    # Add space between words that are stuck together
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()

def read_first_page(pdf_path):
    """Read and print the first page of a PDF file"""
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"Error: File not found at {pdf_path}")
            return

        # Open and read the PDF
        reader = PdfReader(pdf_path)
        
        # Get the first page
        first_page = reader.pages[0]
        
        # Extract and clean the text
        text = first_page.extract_text()
        cleaned_text = clean_text(text)
        
        # Print the cleaned text
        print("\nFirst Page Content:")
        print("=" * 80)
        print(cleaned_text)
        print("=" * 80)
        
        # Print some metadata
        print(f"\nTotal pages in document: {len(reader.pages)}")
        
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "downloads/BackGammon.pdf"
    
    # Read and print the first page
    read_first_page(pdf_path) 