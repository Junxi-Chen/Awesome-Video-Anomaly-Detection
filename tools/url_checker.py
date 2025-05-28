import re
import requests
from bs4 import BeautifulSoup
from collections import Counter


def extract_urls_from_md(md_text):
    return re.findall(r'(?<!\!)\[.*?\]\((https?://[^\s)]+)\)', md_text)


def is_url_valid(url, timeout=10):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LinkChecker/1.0)"
        }
        resp = requests.get(url, timeout=timeout, headers=headers)
        if not resp.ok:
            return False, resp.status_code
        if "text/html" in resp.headers.get("Content-Type", ""):
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string if soup.title else ""
            if "not found" in title.lower() or "404" in title.lower():
                return False, 404
        return True, resp.status_code
    except Exception as e:
        return False, str(e)


def check_markdown_links(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    urls = extract_urls_from_md(content)
    print(f"Found {len(urls)} total links.")

    url_counter = Counter(urls)
    duplicates = [url for url, count in url_counter.items() if count > 1]
    if duplicates:
        print("\nDuplicate links detected:")
        for url in duplicates:
            print(f"- {url} (appears {url_counter[url]} times)")
    else:
        print("\nNo duplicate links found.")

    print("\nChecking link validity:")
    for url in urls:
        ok, status = is_url_valid(url)
        status_str = "✅ OK" if ok else f"❌ Invalid ({status})"
        print(f"{status_str}: {url}")


if __name__ == "__main__":
    check_markdown_links("README.md")
