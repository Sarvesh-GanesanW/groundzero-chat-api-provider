import io
import re
import os
import json
import string
import PyPDF2
import pandas as pd
import csv
import httpx
from typing import List, Dict, Any, Tuple, Optional, Union
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from constants import Constants

# File parsing functions
def parsePdf(fileContent: bytes) -> str:
    """Parse PDF file content and extract text."""
    text = ""
    metadata = {}
    try:
        pdfReader = PyPDF2.PdfReader(io.BytesIO(fileContent))
        metadata = pdfReader.metadata or {}
        for page_num, page in enumerate(pdfReader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text:
                text += f"[Page {page_num}]\n{page_text}\n\n"
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    return text

def parseCsv(fileContent: bytes) -> str:
    """Parse CSV file content and convert to structured text."""
    text = ""
    try:
        decodedContent = fileContent.decode('utf-8', errors='replace')
        csvReader = csv.reader(io.StringIO(decodedContent))
        headers = next(csvReader, [])
        if headers:
            text += " | ".join(headers) + "\n"
            text += "-" * (sum(len(h) for h in headers) + 3 * len(headers)) + "\n"
        
        rows = list(csvReader)
        for row in rows:
            text += " | ".join(row) + "\n"
        
        text += f"\n[Summary: {len(rows)} rows, {len(headers)} columns]\n"
    except Exception as e:
        print(f"Error parsing CSV: {e}")
    return text

def parseExcel(fileContent: bytes) -> str:
    """Parse Excel file content and convert to structured text."""
    text = ""
    try:
        xls = pd.ExcelFile(io.BytesIO(fileContent))
        for sheetName in xls.sheet_names:
            df = pd.read_excel(io.BytesIO(fileContent), sheet_name=sheetName)
            text += f"Sheet: {sheetName} [{df.shape[0]} rows x {df.shape[1]} columns]\n"
            text += df.to_string(index=False) + "\n\n"
    except Exception as e:
        print(f"Error parsing Excel: {e}")
    return text

def parseJson(fileContent: bytes) -> str:
    """Parse JSON file content and convert to formatted text."""
    try:
        decodedContent = fileContent.decode('utf-8', errors='replace')
        data = json.loads(decodedContent)
        return json.dumps(data, indent=2)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return ""

def parseText(fileContent: bytes) -> str:
    """Parse plain text file content."""
    try:
        return fileContent.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Error parsing text file: {e}")
        return ""

def parseHtml(fileContent: bytes) -> str:
    """Parse HTML file content and extract clean text."""
    try:
        decodedContent = fileContent.decode('utf-8', errors='replace')
        soup = BeautifulSoup(decodedContent, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ""

# Web content extraction
async def fetchWebContent(url: str) -> str:
    """Fetch and extract content from a web URL."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.extract()
                return soup.get_text(separator='\n', strip=True)
            elif 'application/pdf' in content_type:
                return parsePdf(response.content)
            elif 'application/json' in content_type:
                return parseJson(response.content)
            else:
                return response.text
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

# Text processing functions
def cleanText(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', text)
    return text.strip()

def extractKeyPhrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract key phrases from text based on frequency."""
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()
    
    # Split into words and count frequency
    words = text.split()
    word_freq = {}
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are'}
    
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_phrases]]

# Text chunking functions
def chunkByWords(text: str, chunk_size: int = Constants.RAG_CHUNK_SIZE, 
               chunk_overlap: int = Constants.RAG_CHUNK_OVERLAP) -> List[str]:
    """Chunk text by words with specified size and overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def chunkBySentences(text: str, max_chunk_size: int = Constants.RAG_CHUNK_SIZE, 
                   chunk_overlap: int = Constants.RAG_CHUNK_OVERLAP) -> List[str]:
    """Chunk text by sentences with approximate max size and overlap."""
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            if chunk_overlap > 0 and current_chunk:
                # Calculate how many sentences to keep for overlap
                overlap_size = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_size = len(s.split())
                    if overlap_size + s_size <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_size
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            else:
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def chunkByParagraphs(text: str, max_chunk_size: int = Constants.RAG_CHUNK_SIZE) -> List[str]:
    """Chunk text by paragraphs with approximate max size."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        paragraph_size = len(paragraph.split())
        
        if current_size + paragraph_size <= max_chunk_size:
            current_chunk.append(paragraph)
            current_size += paragraph_size
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_size = paragraph_size
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

# File type detection and unified parser
def detectFileType(filename: str) -> str:
    """Detect file type from filename extension."""
    extension = os.path.splitext(filename.lower())[1]
    file_types = {
        '.pdf': 'pdf',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json',
        '.txt': 'text',
        '.html': 'html',
        '.htm': 'html',
        '.md': 'text',
        '.rtf': 'text',
        '.xml': 'text',
        '.doc': 'text',
        '.docx': 'text'
    }
    return file_types.get(extension, 'unknown')

def parseFile(fileContent: bytes, filename: str) -> str:
    """Parse file content based on detected file type."""
    file_type = detectFileType(filename)
    
    parsers = {
        'pdf': parsePdf,
        'csv': parseCsv,
        'excel': parseExcel,
        'json': parseJson,
        'text': parseText,
        'html': parseHtml
    }
    
    parser = parsers.get(file_type)
    if parser:
        return parser(fileContent)
    else:
        # Try to parse as text if type is unknown
        return parseText(fileContent)

# Backward compatibility functions
def chunkText(text: str, chunkSize: int = Constants.RAG_CHUNK_SIZE, 
            chunkOverlap: int = Constants.RAG_CHUNK_OVERLAP) -> List[str]:
    """Legacy chunking function for backward compatibility."""
    return chunkByWords(text, chunkSize, chunkOverlap)

# Main RAG processing function
def processDocumentForRag(fileContent: bytes, filename: str, 
                        chunk_method: str = 'words', 
                        clean: bool = True) -> List[str]:
    """Process document for RAG with flexible chunking options."""
    # Extract text from document
    text = parseFile(fileContent, filename)
    
    # Clean text if requested
    if clean:
        text = cleanText(text)
    
    # Chunk the text based on requested method
    if chunk_method == 'sentences':
        return chunkBySentences(text)
    elif chunk_method == 'paragraphs':
        return chunkByParagraphs(text)
    else:  # Default to words
        return chunkByWords(text)