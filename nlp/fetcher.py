import os
from typing import Optional

from googleapiclient.discovery import build

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")


def fetch_comments(video_id: str, max_results: int = 500) -> list[dict]:
    if not YOUTUBE_API_KEY:
        raise RuntimeError(
            "Set the YOUTUBE_API_KEY environment variable to enable live fetching."
        )

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    comments: list[dict] = []
    next_page_token: Optional[str] = None

    while len(comments) < max_results:
        batch_size = min(100, max_results - len(comments))
        resp = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                order="relevance",
                textFormat="plainText",
                pageToken=next_page_token,
            )
            .execute()
        )

        for item in resp.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "author": snippet.get("authorDisplayName", "Anonymous"),
                    "text": snippet.get("textDisplay", ""),
                    "likes": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt", ""),
                    "updated_at": snippet.get("updatedAt", ""),
                }
            )

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def fetch_video_title(video_id: str) -> str:
    if not YOUTUBE_API_KEY:
        return "Unknown Video"
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    resp = youtube.videos().list(part="snippet", id=video_id).execute()
    items = resp.get("items", [])
    if items:
        return items[0]["snippet"]["title"]
    return "Unknown Video"
