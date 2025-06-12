import os
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import PyPDF2
from fpdf import FPDF
from datetime import datetime

def get_blog_posts():
    base_url = "https://thegammonpress.com"
    categories = [
        "/category/backgammon-problems/",
        "/category/backgammon-tips-for-beginners/",
        "/category/learning-backgammon/",
        "/category/backgammon-generally/",
        "/category/backgammon-problems-early-game/",
        "/"  # Main blog page
    ]
    
    all_posts = set()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Regular expression to match blog post URLs
    # Matches URLs like: /mar-01-2021-priming-games-escape-or-attack/
    blog_pattern = re.compile(r'/([a-z]{3}-\d{2}-\d{4}-[^/]+)/$')
    
    for category in categories:
        page_num = 1
        while True:
            url = f"{base_url}{category}"
            if page_num > 1:
                url = f"{url}page/{page_num}/"
            
            print(f"\nFetching: {url}")
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all links on the page
                all_links = soup.find_all('a', href=True)
                print(f"\nFound {len(all_links)} total links on page")
                
                # Look for blog post links
                for link in all_links:
                    href = link['href']
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)
                    
                    # Check if the URL matches our blog post pattern
                    if blog_pattern.search(href):
                        print(f"Found blog post: {href}")
                        all_posts.add(href)
                
                # Check if there's a next page
                next_page = soup.find('a', class_='next')
                if not next_page:
                    print("\nNo next page found")
                    break
                    
                page_num += 1
                time.sleep(1)  # Be nice to the server
                
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                break
    
    return all_posts

def download_post(url, output_dir):
    try:
        print(f"\nAttempting to download: {url}")
        
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Initialize the Chrome driver
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Load the page
            driver.get(url)
            
            # Wait for the content to load (wait for the entry-content div)
            wait = WebDriverWait(driver, 10)
            content_div = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'entry-content')))
            
            # Get the page source after JavaScript has loaded
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Extract title
            title = soup.find('h1', class_='entry-title')
            print(f"\nFound title with class 'entry-title': {title is not None}")
            if not title:
                title = soup.find('h1')
                print(f"Found regular h1 title: {title is not None}")
            if not title:
                # Extract title from URL if we can't find it in the HTML
                title = url.split('/')[-2]  # Get the last part of the URL
                title = title.split('-', 3)[-1]  # Remove the date part
                title = title.replace('-', ' ').title()  # Convert to title case
                print(f"Extracted title from URL: {title}")
            else:
                title = title.text.strip()
                print(f"Using title from HTML: {title}")
            
            # Clean title for filename
            title = re.sub(r'[^\w\s-]', '', title)
            title = re.sub(r'[-\s]+', '-', title)
            print(f"Cleaned title for filename: {title}")
            
            # Extract content
            content = soup.find('div', class_='entry-content')
            if content:
                # Remove unwanted elements
                for element in content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                    element.decompose()
                
                # Get all paragraphs and other text content
                paragraphs = []
                
                # Get the main text content
                for p in content.find_all(['p', 'h2', 'h3', 'h4', 'ul', 'ol']):
                    text = p.get_text(strip=True)
                    if text:  # Only add non-empty paragraphs
                        paragraphs.append(text)
                        print(f"\nFound paragraph: {text[:100]}...")
                
                # Join all paragraphs with newlines
                text = '\n\n'.join(paragraphs)
                print(f"\nExtracted text content length: {len(text)} characters")
                
                if text.strip():  # Only save if we found actual content
                    # Create filename
                    filename = f"{title}.txt"
                    filepath = os.path.join(output_dir, filename)
                    print(f"Will save to: {filepath}")
                    
                    # Save to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {title}\n")
                        f.write(f"URL: {url}\n")
                        f.write("\nContent:\n")
                        f.write(text)
                    
                    print(f"Successfully saved post to {filepath}")
                    return True
                else:
                    print("No text content found in the main content area")
            else:
                print("No content found in the page")
                print("\nPage structure:")
                print(soup.prettify()[:1000])  # Print first 1000 chars of formatted HTML
                return False
                
        finally:
            # Always close the driver
            driver.quit()
            
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def save_as_pdf(post_data, output_dir):
    """Save the blog post as a PDF"""
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(0, 10, post_data['title'])
    
    # Add date
    pdf.set_font('Arial', 'I', 12)
    pdf.multi_cell(0, 10, post_data['date'])
    
    # Add URL
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, f"Source: {post_data['url']}")
    
    # Add content
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, post_data['content'])
    
    # Create filename from date and title
    date_str = datetime.strptime(post_data['date'], '%b %d, %Y').strftime('%Y-%m-%d')
    filename = f"{date_str}_{post_data['title'].replace(' ', '_')[:50]}.pdf"
    filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
    
    # Save the PDF
    pdf_path = os.path.join(output_dir, filename)
    pdf.output(pdf_path)
    return pdf_path

def main():
    # Create output directory
    output_dir = "gammon_press_posts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all blog posts
    print("Finding blog posts...")
    posts = get_blog_posts()
    print(f"\nFound {len(posts)} unique blog posts")
    
    # Download each post
    print("\nDownloading posts...")
    downloaded = 0
    for url in posts:
        if download_post(url, output_dir):
            downloaded += 1
            print(f"Downloaded: {url}")
        time.sleep(1)  # Be nice to the server
    
    print(f"\nDownloaded {downloaded} posts to {output_dir}/")

if __name__ == "__main__":
    main() 