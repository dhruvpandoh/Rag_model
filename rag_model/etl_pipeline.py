#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import pymongo
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi
from clearml import Task
import logging
from urllib.parse import urljoin, urlparse
import os

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ClearML Task Initialization
task = Task.init(project_name="RAG_ETL_Pipeline", task_name="ROS2_ETL_Pipeline")

# MongoDB Connection
db_string = "mongodb+srv://and8995:Aniks777@cluster0.4voet.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(db_string)
db = client["ros2_rag"]
raw_data_collection = db["raw_data"]
youtube_collection = db["youtube_data"]

def is_valid_url(url, base_domain):
    try:
        parsed = urlparse(url)
        return base_domain in parsed.netloc
    except Exception as e:
        logging.error(f"Error parsing URL: {e}")
        return False

def scrape_page(url, visited_urls, base_domain):
    if url in visited_urls:
        return None, set()
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
        
        text_content = soup.get_text(separator='\n', strip=True)
        
        code_snippets = [code.get_text(strip=True) for code in soup.find_all(['code', 'pre']) if code.get_text(strip=True)]
        
        new_urls = {
            urljoin(url, link.get('href'))
            for link in soup.find_all('a', href=True)
            if is_valid_url(urljoin(url, link.get('href')), base_domain)
        }
        
        data = {
            'url': url,
            'text_content': text_content,
            'code_snippets': code_snippets
        }
        
        return data, new_urls
        
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return None, set()

def crawl_website(start_url, max_pages=200):
    base_domain = urlparse(start_url).netloc
    visited_urls = set()
    to_visit = {start_url}
    collected_data = []
    
    while to_visit and len(visited_urls) < max_pages:
        current_url = to_visit.pop()
        
        if current_url in visited_urls:
            continue
            
        data, new_urls = scrape_page(current_url, visited_urls, base_domain)
        if data:
            collected_data.append(data)
            visited_urls.add(current_url)
            to_visit.update(new_urls - visited_urls)
            logging.info(f"Scraped {current_url}")
            time.sleep(random.uniform(0.5, 1.5))
        else:
            visited_urls.add(current_url)
    
    return collected_data

def ingest_documentation(start_url, collection, subdomain, max_pages=50):
    try:
        logging.info(f"Starting crawl for {subdomain}: {start_url}")
        data_list = crawl_website(start_url, max_pages)
        
        if data_list:
            for document in data_list:
                document['source'] = subdomain
                document['metadata'] = {
                    'type': 'Documentation',
                    'scraped_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            collection.insert_many(data_list)
            logging.info(f"Stored {len(data_list)} pages from {subdomain}")
        else:
            logging.warning(f"No data scraped from {subdomain}")
            
    except Exception as e:
        logging.error(f"Error ingesting documentation from {subdomain}: {str(e)}")

def fetch_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        logging.error(f"Failed to fetch transcript for {video_id}: {e}")
        return ""

def ingest_youtube_transcripts(video_ids, collection):
    for video_id in video_ids:
        transcript = fetch_youtube_transcript(video_id)
        if transcript:
            document = {
                'source': 'youtube',
                'video_id': video_id,
                'content': transcript,
                'metadata': {
                    'type': 'YouTube Transcript',
                    'fetched_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            try:
                collection.insert_one(document)
                logging.info(f"Ingested YouTube transcript: {video_id}")
            except Exception as e:
                logging.error(f"Error storing YouTube transcript {video_id}: {e}")
            time.sleep(random.uniform(0.5, 1.5))
        else:
            logging.warning(f"No transcript found for {video_id}")

def etl_pipeline():
    sources = {
        "ROS.org": "https://www.ros.org/",
        "ROS2 Documentation": "https://docs.ros.org/en/foxy/",
        "Nav2 Documentation": "https://docs.nav2.org/",
        "MoveIt": "https://moveit.ai/",
        "Gazebo Documentation": "https://gazebosim.org/docs/all/getstarted/"
    }

    # Crawl and ingest documentation
    for subdomain, base_url in sources.items():
        ingest_documentation(base_url, raw_data_collection, subdomain, max_pages=50)

    # YouTube transcripts
    youtube_video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0"]
    ingest_youtube_transcripts(youtube_video_ids, youtube_collection)

if __name__ == "__main__":
    etl_pipeline()
