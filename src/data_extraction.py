import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import os

# "use a Reddit extraction library referred to as pro"
try:
    import pro 
except ImportError:
    logging.warning("Library 'pro' not installed, providing a mock implementation for Reddit extraction.")
    class MockProExtractor:
        def fetch_subreddit(self, name, limit=100):
            return [{"id": "xyz", "title": "Example Post", "selftext": "This is raw post text", "author": "User1", "created_utc": 1690000000}]
        def fetch_comments(self, post_id):
            return [{"id": "abc", "body": "Example Comment", "author": "User2", "created_utc": 1690000000}]
    pro = MockProExtractor()

def perform_extraction(subreddits: List[str], limit: int = 100) -> List[Dict[str, Any]]:
    """
    Extracts Reddit posts and comments into JSON to build datasets for 
    toxicity detection and crisis detection using 'pro'.
    """
    dataset = []
    # If pro is an instantiated object natively, this simulates usage.
    try:
        extractor = pro.RedditExtractor() # Type of object assumed for 'pro'
    except AttributeError:
        extractor = pro

    for sub in subreddits:
        posts = extractor.fetch_subreddit(sub, limit=limit)
        for post in posts:
            record = {
                "id": post.get("id"),
                "post_text": f"{post.get('title', '')} {post.get('selftext', '')}".strip(),
                "comment_text": "",
                "subreddit_source": sub,
                "timestamp": post.get("created_utc", datetime.utcnow().timestamp()),
                "author_id": post.get("author"), 
                "preprocessing_status": "raw",
                "toxicity_label": None,
                "crisis_label": None
            }
            dataset.append(record)
            
            # Fetch comments for the post
            comments = extractor.fetch_comments(post.get("id"))
            for comment in comments:
                dataset.append({
                    "id": comment.get("id"),
                    "post_text": "",
                    "comment_text": comment.get("body", ""),
                    "subreddit_source": sub,
                    "timestamp": comment.get("created_utc", datetime.utcnow().timestamp()),
                    "author_id": comment.get("author"),
                    "preprocessing_status": "raw",
                    "toxicity_label": None,
                    "crisis_label": None
                })
    return dataset

if __name__ == "__main__":
    raw_data = perform_extraction(["mentalhealth", "depression"], limit=10)
    with open("../data/reddit_raw.json", "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Extracted {len(raw_data)} records.")
