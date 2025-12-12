import json
import os
import re
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import requests
from PIL import Image
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

# Entry page listing meme templates with "Add Caption" links
START_URL = "https://imgflip.com/memetemplates?sort=top-all-time"

# Output locations (written at repo root by default)
OUTFILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memes.json")
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memes_images")

# basic logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("imgflip_scraper")


def normalize_url(base: str, candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    abs_url = urljoin(base, candidate)
    if abs_url.startswith("data:") or abs_url.startswith("#"):
        return None
    return abs_url


def get_soup(url: str) -> BeautifulSoup:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; imgflip-scraper/1.0; +https://example.com/bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    logger.debug(f"GET {url}")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def get_rendered_soup(
    url: str,
    wait_selectors: Optional[List[str]] = None,
    wait_time: float = 10.0,
    max_wait_per_selector: float = 15.0,
    retries: int = 2,
    retry_backoff_seconds: float = 1.0,
) -> Optional[BeautifulSoup]:
    """Render a URL with a headless browser and return BeautifulSoup of the final DOM.

    Attempts to use Selenium with headless Chrome if available. If Selenium or a
    compatible driver is not available, returns None.
    """

    def render_once() -> Optional[str]:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,2200")
        options.page_load_strategy = 'none'
        # Speed hints; HTML is enough to extract src/style
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument(
            "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        )

        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            logger.debug(f"Rendering (selenium) GET {url}")
            driver.get(url)

            # Single JS-based wait for ANY of the selectors instead of sequential waits
            combined = None
            if wait_selectors and len(wait_selectors) > 0:
                combined = ",".join(wait_selectors)
            deadline = time.time() + max(1.0, max_wait_per_selector)
            while time.time() < deadline:
                try:
                    if combined:
                        found = driver.execute_script(
                            "return document.querySelector(arguments[0]) !== null;",
                            combined,
                        )
                        if bool(found):
                            break
                    else:
                        # No target selectors provided; just a minimal wait
                        if time.time() >= deadline - 0.1:
                            break
                except Exception:
                    # Ignore transient execution errors during early page load
                    pass
                time.sleep(0.2)

            return driver.page_source or ""
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass

    last_error: Optional[str] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            html = render_once()
            if html and html.strip():
                return BeautifulSoup(html, "html.parser")
            last_error = "empty html"
        except Exception as e:  # pragma: no cover - defensive
            last_error = str(e)
        if attempt < max(1, retries) + 1:
            time.sleep(retry_backoff_seconds * attempt)

    logger.info(f"Dynamic rendering failed (selenium): {last_error}")
    return None


def _parse_preview_width_from_style(style: Optional[str]) -> Optional[float]:
    if not style:
        return None
    # Prefer explicit width first
    m = re.search(r"width:\s*([^;]+)", style)
    if m:
        px = parse_px(m.group(1))
        if px is not None:
            return px
    # Fallback to max-width if present
    m = re.search(r"max-width:\s*([^;]+)", style)
    if m:
        px = parse_px(m.group(1))
        if px is not None:
            return px
    return None

def update_positions_for_meme(meme_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add updated_positions to each text_options item using ratio real_width/preview_width.

    Keeps existing structure and adds an 'updated_positions' dict with left/top/right/bottom
    measured in actual image pixels.
    """
    url = meme_data.get("url")
    page_data = extract_page_data(url)
    _, orig_w, _ = fetch_image_and_size(page_data.get("image_url"))

    ratio = orig_w / float(page_data.get("preview_width"))

    for text_option, meme_text_option in zip(page_data.get("boxes", []), meme_data.get("text_options", [])):
        updated_position = {
            "left": round(text_option["left"] * ratio),
            "top": round(text_option["top"] * ratio), 
            "width": round(text_option["width"] * ratio),
            "height": round(text_option["height"] * ratio)
        }
        meme_text_option["updated_position"] = updated_position
    meme_data['has_updated_positions'] = True
    return meme_data


def update_positions_bulk(concurrency: int = 10) -> None:
    """Update positions concurrently in batches and write after each batch.

    - Spawns up to `concurrency` workers (default 10).
    - Only processes records without 'has_updated_positions'.
    - Writes the full memes.json after each completed batch (atomic write).
    """
    memes: Dict[str, Any] = load_existing_memes()
    if not memes:
        logger.info("No existing memes found to update.")
        return

    keys_to_update: List[str] = [k for k, v in memes.items() if not v.get("has_updated_positions")]
    if not keys_to_update:
        logger.info("All memes already have updated positions.")
        return

    logger.info(f"Updating positions for {len(keys_to_update)} memes with concurrency={concurrency}.")

    def _update_one(key: str, meme: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # Work on a copy to avoid mutating shared objects across threads
        local = copy.deepcopy(meme)
        updated = update_positions_for_meme(local)
        return key, updated

    for start in range(0, len(keys_to_update), concurrency):
        batch_keys = keys_to_update[start:start + concurrency]
        results: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_update_one, key, memes[key]) for key in batch_keys]
            for fut in as_completed(futures):
                try:
                    key, updated_meme = fut.result()
                    results[key] = updated_meme
                except Exception as e:  # pragma: no cover - defensive
                    # Keep original meme on failure; log and continue
                    key_idx = futures.index(fut) if fut in futures else None
                    key_fallback = batch_keys[key_idx] if key_idx is not None else "unknown"
                    logger.exception(f"Failed to update positions for meme '{key_fallback}': {e}")
                    if key_idx is not None:
                        results[key_fallback] = memes[key_fallback]

        # Merge results and write atomically after each batch
        memes.update(results)
        write_memes(memes)
        logger.info(f"Wrote batch {(start // concurrency) + 1} ({len(batch_keys)} items) to {OUTFILE}")

    logger.info("Finished updating positions for all pending memes.")
  
def parse_px(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = re.search(r"(-?\d+\.?\d*)px", value)
    return float(m.group(1)) if m else None


def parse_dimension_to_int(value: Optional[str]) -> Optional[int]:
    """Parse dimension strings like '756', '756.6', or '756px' into an int.

    Returns None if no numeric value can be parsed.
    """
    if value is None:
        return None
    s = str(value).strip()
    # Try 'px' style first
    px = parse_px(s)
    if px is not None:
        return int(round(px))
    # Try plain int/float
    try:
        return int(s)
    except Exception:
        try:
            return int(round(float(s)))
        except Exception:
            m = re.search(r"(-?\d+\.?\d*)", s)
            return int(round(float(m.group(1)))) if m else None


def collect_generator_links(max_pages: Optional[int] = None) -> List[str]:
    generator_urls: List[str] = []
    seen: set[str] = set()

    page_index = 1
    while True:
        page_url = START_URL if page_index == 1 else f"{START_URL}?page={page_index}"
        logger.debug(f"Scanning template page {page_index}: {page_url} (max_pages={max_pages or '∞'})")
        soup = get_soup(page_url)

        anchors = soup.find_all("a", class_=lambda c: c and "mt-caption" in c)
        logger.debug(f"Found {len(anchors)} anchors with class mt-caption on page {page_index}")
        page_links: List[str] = []
        for a in anchors:
            href = a.get("href")
            text = (a.get_text(strip=True) or "").lower()
            if "add caption" not in text:
                # Some sites may not include the text in the same node; accept any mt-caption link
                pass
            abs_url = normalize_url(page_url, href)
            if abs_url and "/memegenerator/" in abs_url and abs_url not in seen:
                seen.add(abs_url)
                page_links.append(abs_url)

        logger.debug(f"Page {page_index}: collected {len(page_links)} generator links (total so far {len(generator_urls) + len(page_links)})")

        if not page_links:
            logger.debug("No new generator links found on this page; stopping pagination.")
            break

        generator_urls.extend(page_links)
        if max_pages is not None and page_index >= max_pages:
            logger.debug(f"Reached max_pages={max_pages}; stopping pagination.")
            break
        page_index += 1

    return generator_urls


def fetch_image_and_size(image_url: str) -> Tuple[bytes, int, int]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; imgflip-scraper/1.0; +https://example.com/bot)",
        "Referer": "https://imgflip.com/",
    }
    logger.debug(f"Downloading image: {image_url}")
    resp = requests.get(image_url, headers=headers, timeout=60)
    resp.raise_for_status()
    content = resp.content
    with Image.open(BytesIO(content)) as im:
        width, height = im.size
    logger.info(f"Downloaded image {image_url} ({width}x{height})")
    return content, width, height


def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9._-]", "", name)


def extract_page_data(url: str) -> Optional[Dict[str, Any]]:
    logger.info(f"scraping this page: {url}")
    soup = get_rendered_soup(
        url,
        wait_selectors=[".m-preview", ".mm-img", "canvas.mm-canv"],
        wait_time=6.0,
        max_wait_per_selector=15.0,
        retries=2,
        retry_backoff_seconds=1.0,
    )
    if soup is None:
        logger.warning("Dynamic rendering unavailable or failed; aborting page extraction")
        return None

    # Meme name from URL path
    path_last = urlparse(url).path.rstrip("/").split("/")[-1]
    meme_name = path_last

    preview = soup.find("div", class_=lambda c: c and "m-preview" in c)
    # If preview not found on static HTML, try dynamic rendering
    if not preview:
        dyn_soup = get_rendered_soup(url, wait_selectors=[".m-preview", ".mm-img"], wait_time=4.0)
        if dyn_soup is not None:
            soup = dyn_soup
            preview = soup.find("div", class_=lambda c: c and "m-preview" in c)
    if not preview:
        logger.info("Preview div .m-preview not found")
        return None

    # Try to find the main image source inside preview
    img_tag = preview.find("img", class_=lambda c: c and "mm-img" in c)
    img_src = img_tag.get("src") if img_tag else None
    if not img_src:
        # Fallback: sometimes the image may be set as background on a div
        bg_div = preview.find("div", class_=lambda c: c and "mm-img" in c)
        style = bg_div.get("style") if bg_div else None
        if style:
            m = re.search(r"url\(([^)]+)\)", style)
            img_src = m.group(1).strip('"\'') if m else None
    # If still missing img_src, attempt dynamic rendering as a second-chance
    if not img_src:
        dyn_soup = get_rendered_soup(url, wait_selectors=[".m-preview", ".mm-img"], wait_time=4.0)
        if dyn_soup is not None:
            preview = dyn_soup.find("div", class_=lambda c: c and "m-preview" in c)
            if preview:
                img_tag = preview.find("img", class_=lambda c: c and "mm-img" in c)
                img_src = img_tag.get("src") if img_tag else None
                if not img_src:
                    bg_div = preview.find("div", class_=lambda c: c and "mm-img" in c)
                    style = bg_div.get("style") if bg_div else None
                    if style:
                        m = re.search(r"url\(([^)]+)\)", style)
                        img_src = m.group(1).strip('"\'') if m else None

    if not img_src:
        logger.info("Could not locate preview image src on page")
        return None

    image_url = normalize_url(url, img_src)
    if not image_url:
        logger.info("Preview image URL normalization failed")
        return None

    # Canvas sizing
    canvas = preview.find("canvas", class_=lambda c: c and "mm-canv" in c)
    canvas_width = parse_dimension_to_int(canvas.get("width")) if canvas else None
    canvas_height = parse_dimension_to_int(canvas.get("height")) if canvas else None
    logger.debug(f"Canvas size: {canvas_width}x{canvas_height}")

    preview_width = _parse_preview_width_from_style(preview.get("style")) if preview else None

    # Drag boxes (text areas)
    drag_boxes = preview.find_all("div", class_=lambda c: c and "drag-box" in c and "off" in c)
    boxes: List[Dict[str, float]] = []
    for db in drag_boxes:
        style = db.get("style") or ""
        left = parse_px(re.search(r"left:\s*([^;]+)", style).group(1) if re.search(r"left:\s*([^;]+)", style) else None)
        top = parse_px(re.search(r"top:\s*([^;]+)", style).group(1) if re.search(r"top:\s*([^;]+)", style) else None)
        width = parse_px(re.search(r"width:\s*([^;]+)", style).group(1) if re.search(r"width:\s*([^;]+)", style) else None)
        height = parse_px(re.search(r"height:\s*([^;]+)", style).group(1) if re.search(r"height:\s*([^;]+)", style) else None)
        if None not in (left, top, width, height):
            boxes.append({"left": left, "top": top, "width": width, "height": height})
    logger.debug(f"Found {len(boxes)} drag-box positions")

    return {
        "meme_name": meme_name,
        "image_url": image_url,
        "preview_width": preview_width,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "boxes": boxes,
    }


def scale_boxes_to_original(boxes: List[Dict[str, float]], canvas_size: Tuple[Optional[int], Optional[int]], original_size: Tuple[int, int]) -> List[Dict[str, int]]:
    canvas_w, canvas_h = canvas_size
    orig_w, orig_h = original_size
    if not canvas_w or not canvas_h or canvas_w <= 0 or canvas_h <= 0:
        # If canvas missing, return integer-cast boxes without scaling
        return [
            {
                "left": int(round(b["left"])),
                "top": int(round(b["top"])),
                "width": int(round(b["width"])),
                "height": int(round(b["height"]))
            }
            for b in boxes
        ]

    scale_x = orig_w / float(canvas_w)
    scale_y = orig_h / float(canvas_h)

    scaled: List[Dict[str, int]] = []
    for b in boxes:
        scaled.append(
            {
                "left": int(round(b["left"] * scale_x)),
                "top": int(round(b["top"] * scale_y)),
                "width": int(round(b["width"] * scale_x)),
                "height": int(round(b["height"] * scale_y)),
            }
        )
    return scaled


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_existing_memes() -> Dict[str, Any]:
    if not os.path.exists(OUTFILE):
        return {}
    try:
        with open(OUTFILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning(f"Failed to read existing {OUTFILE}: {e}")
    return {}


def write_memes(memes: Dict[str, Any]) -> None:
    tmp = OUTFILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(memes, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTFILE)


def scrape_memes(max_pages: int = 1) -> Dict[str, Any]:
    ensure_dir(IMAGES_DIR)

    memes: Dict[str, Any] = load_existing_memes()
    if memes:
        logger.info(f"Loaded {len(memes)} existing memes from {OUTFILE}")

    generator_links = collect_generator_links(max_pages=max_pages)
    logger.info(f"Total generator links to process: {len(generator_links)}")
    for idx, gen_url in enumerate(generator_links, start=1):
        try:
            meme_key = urlparse(gen_url).path.rstrip("/").split("/")[-1]
            meme_name = meme_key

            if meme_key in memes:
                # logger.info(f"Meme already exists, skipping this page: {gen_url}")
                continue

            page_data = extract_page_data(gen_url)
            if not page_data:
                logger.info(f"No Data found, skipping this page: {gen_url}")
                continue

            image_url = page_data["image_url"]
            canvas_size = (page_data.get("canvas_width"), page_data.get("canvas_height"))


            # Download image and determine original size
            content, orig_w, orig_h = fetch_image_and_size(image_url)

            # Save image to disk
            file_base = safe_filename(meme_key) or "image"
            ext = os.path.splitext(urlparse(image_url).path)[1] or ".jpg"
            filename = f"{file_base}{ext}"
            file_path = os.path.join(IMAGES_DIR, filename)
            # Skip saving file if already exists with non-zero size
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"Image already exists, skipping save: {file_path}")
            else:
                with open(file_path, "wb") as f:
                    f.write(content)
                logger.debug(f"Saved image to {file_path}")

            # Scale drag-box positions
            scaled_boxes = scale_boxes_to_original(
                page_data.get("boxes", []),
                canvas_size,
                (orig_w, orig_h),
            )

            # Build text options structure
            text_options = [
                {
                    "position": b,
                    "description": "",
                }
                for b in scaled_boxes
            ]

            # Confirm name also based on image filename (without extension)
            img_last = os.path.basename(urlparse(image_url).path)
            img_stem = os.path.splitext(img_last)[0]

            record = {
                "name": meme_key,
                "url": gen_url,
                "image_url": image_url,
                "filename": filename,
                "width": orig_w,
                "height": orig_h,
                "image_name_from_src": img_stem,
                "text_options": text_options,
            }
            memes[meme_key] = record
            write_memes(memes)
            logger.info(f"scraped and saved: {gen_url}")
        except Exception as e:
            logger.exception(f"Error processing {gen_url}: {e}")
            continue

    return memes


def main() -> None:

    update_positions_bulk()
    return

    logger.info("Starting scrape...")
    memes = scrape_memes(max_pages=args.max_pages)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(memes, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {len(memes)} memes to {OUTFILE} and images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
