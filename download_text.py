import requests
import os
import hashlib
from PyPDF2 import PdfReader
import re

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file to check for duplicates"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None

def file_already_downloaded(file_name, downloads_dir='downloads'):
    """Check if a file with the same name already exists in downloads folder"""
    file_path = os.path.join(downloads_dir, file_name)
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def clean_text(text):
    """Clean extracted text by adding spaces between words and removing excessive whitespace"""
    # Add space between words that are stuck together
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()

def download_pdf(url, file_name=None, downloads_dir='downloads'):
    """
    Download a PDF from URL if it doesn't already exist
    
    Args:
        url (str): Direct URL to the PDF
        file_name (str): Optional custom filename. If None, extracts from URL
        downloads_dir (str): Directory to save downloads
    """
    
    # Extract filename from URL if not provided
    if file_name is None:
        file_name = url.split('/')[-1]
        if not file_name.endswith('.pdf'):
            file_name += '.pdf'
    
    # Ensure filename ends with .pdf
    if not file_name.endswith('.pdf'):
        file_name += '.pdf'
    
    # Create downloads directory if it doesn't exist
    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)
        print(f"Created directory: {downloads_dir}")
    
    # Check if file already exists
    if file_already_downloaded(file_name, downloads_dir):
        print(f"File '{file_name}' already exists in {downloads_dir}. Skipping download.")
        
        # Still verify the existing file
        file_path = os.path.join(downloads_dir, file_name)
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            file_size = os.path.getsize(file_path)
            print(f"Existing file verified: {num_pages} pages, {file_size} bytes")
        except Exception as e:
            print(f"Warning: Existing file appears corrupted: {e}")
            print("Will re-download...")
        else:
            return file_path
    
    # File path to save the PDF
    file_path = os.path.join(downloads_dir, file_name)
    
    try:
        # Send a GET request to the URL with proper headers
        print(f"Downloading PDF: {file_name}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf',
            'Accept-Encoding': 'identity',  # Prevent compression
            'Connection': 'keep-alive'
        }
        
        # Download the file with progress tracking
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress tracking
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='')
        
        print("\nDownload completed!")
        
        # Verify the downloaded file
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")
            
            # Try to read the PDF
            try:
                reader = PdfReader(file_path)
                num_pages = len(reader.pages)
                print(f"Successfully read PDF with {num_pages} pages")
                
                # Print first page text as a sample
                first_page = reader.pages[0]
                text = first_page.extract_text()
                cleaned_text = clean_text(text)
                print("\nFirst page preview:")
                print("-" * 50)
                print(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)
                print("-" * 50)
                
            except Exception as e:
                print(f"Error reading PDF: {e}")
        else:
            print("Error: File was not created")
            return None
        
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def download_multiple_pdfs(pdf_list):
    """
    Download multiple PDFs from a list of URLs or (URL, filename) tuples
    
    Args:
        pdf_list: List of URLs or tuples of (URL, filename)
    """
    downloaded_files = []
    
    for item in pdf_list:
        if isinstance(item, tuple):
            url, filename = item
            result = download_pdf(url, filename)
        else:
            url = item
            result = download_pdf(url)
        
        if result:
            downloaded_files.append(result)
        
        print("\n" + "="*60 + "\n")
    
    print(f"Download summary: {len(downloaded_files)} files processed")
    for file_path in downloaded_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    # List of PDFs to download
    # You can add more URLs here, either as strings or (URL, filename) tuples
    pdfs_to_download = [
        "https://dn790006.ca.archive.org/0/items/gnubg_vs_mamoun_html_source_files/GNUBackgammon_vs_Mamoun.pdf",
        "https://www.edwardothorp.com/wp-content/uploads/2016/11/BackGammon.pdf",
        # Add more URLs here as needed
        # ("https://example.com/another.pdf", "custom_name.pdf"),  # Custom filename example
    ]
    
    # Download all PDFs
    download_multiple_pdfs(pdfs_to_download)