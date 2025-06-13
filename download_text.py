import requests
import os
import hashlib
from PyPDF2 import PdfReader
import re
import shutil
from PyPDF2 import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

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
    # Clean up the text like in text_transformer.py
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Keep only basic punctuation
    return text.strip()

def process_text_file(source_path, downloads_dir='/root/Backgammon/downloads'):
    """
    Process a text file and save it as a PDF in the downloads directory
    
    Args:
        source_path (str): Path to the text file
        downloads_dir (str): Directory to save processed files
    """
    try:
        # Create downloads directory if it doesn't exist
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)
            print(f"Created directory: {downloads_dir}")
        
        # Get the filename from the source path
        file_name = os.path.basename(source_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Read the text file
        print(f"\nProcessing text file: {file_name}")
        print(f"Source path: {source_path}")
        
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying with different encoding...")
            with open(source_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Clean the text
        cleaned_content = clean_text(content)
        
        # Create a PDF using reportlab
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Set font and size
        c.setFont("Helvetica", 12)
        
        # Split text into lines that fit the page width
        lines = []
        words = cleaned_content.split()
        current_line = []
        for word in words:
            current_line.append(word)
            line_width = c.stringWidth(' '.join(current_line), "Helvetica", 12)
            if line_width > width - 100:  # Leave margins
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Write text to PDF
        y = height - 50  # Start from top with margin
        for line in lines:
            if y < 50:  # If we're too close to bottom, start new page
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - 50
            c.drawString(50, y, line)
            y -= 15  # Move down for next line
        
        c.save()
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        
        # Save to downloads directory with .pdf extension
        output_path = os.path.join(downloads_dir, f"{base_name}.pdf")
        with open(output_path, 'wb') as f:  # Note: 'wb' for binary write
            f.write(pdf_content)
        
        print(f"\nSuccessfully processed and saved: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
        print(f"Full path to saved file: {os.path.abspath(output_path)}")
        
        # Print preview
        print("\nFile preview:")
        print("-" * 50)
        print(cleaned_content[:500] + "..." if len(cleaned_content) > 500 else cleaned_content)
        print("-" * 50)
        
        return output_path
        
    except Exception as e:
        print(f"\nError processing text file {source_path}: {e}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
        return None

def process_external_drive(drive_path, downloads_dir='downloads'):
    """
    Process all text files from the external drive
    
    Args:
        drive_path (str): Path to the external drive
        downloads_dir (str): Directory to save processed files
    """
    processed_files = []
    
    try:
        print(f"\nChecking external drive path: {drive_path}")
        print(f"Path exists: {os.path.exists(drive_path)}")
        print(f"Path is directory: {os.path.isdir(drive_path)}")
        
        if not os.path.exists(drive_path):
            print(f"Error: Drive path does not exist: {drive_path}")
            # Try alternative path formats
            alt_paths = [
                drive_path.replace('/', '\\'),  # Windows backslash
                drive_path.replace('\\', '/'),  # Forward slash
                os.path.abspath(drive_path),    # Absolute path
            ]
            print("\nTrying alternative paths:")
            for alt_path in alt_paths:
                print(f"Checking: {alt_path}")
                if os.path.exists(alt_path):
                    print(f"Found working path: {alt_path}")
                    drive_path = alt_path
                    break
            else:
                print("No alternative paths worked. Please check the drive path.")
                return
        
        # List contents of the drive
        print("\nContents of drive:")
        try:
            contents = os.listdir(drive_path)
            for item in contents:
                full_path = os.path.join(drive_path, item)
                print(f"  - {item} ({'directory' if os.path.isdir(full_path) else 'file'})")
        except Exception as e:
            print(f"Error listing drive contents: {e}")
        
        # Walk through all files in the drive path
        print("\nSearching for text files...")
        found_files = False
        for root, dirs, files in os.walk(drive_path):
            print(f"\nChecking directory: {root}")
            print(f"Found {len(files)} files")
            
            for file in files:
                if file.endswith('.txt'):
                    found_files = True
                    source_path = os.path.join(root, file)
                    print(f"\nFound text file: {source_path}")
                    result = process_text_file(source_path, downloads_dir)
                    if result:
                        processed_files.append(result)
                    print("\n" + "="*60 + "\n")
        
        if not found_files:
            print("\nNo .txt files found in the drive path.")
            print("Please check if:")
            print("1. The drive path is correct")
            print("2. There are .txt files in the drive")
            print("3. You have permission to access the files")
    
    except Exception as e:
        print(f"\nError accessing external drive {drive_path}: {e}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
    
    print(f"\nProcessing summary: {len(processed_files)} files processed")
    for file_path in processed_files:
        print(f"  - {file_path}")

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
    # Example usage
    pdf_list = [
        "https://dn790006.ca.archive.org/0/items/gnubg_vs_mamoun_html_source_files/GNUBackgammon_vs_Mamoun.pdf",
        # "https://www.edwardothorp.com/wp-content/uploads/2016/11/BackGammon.pdf",
        # "https://www.mathnet.ru/links/658c6c904d7271d98d0427836714eed2/cgtm255.pdf",
    ]
    download_multiple_pdfs(pdf_list)