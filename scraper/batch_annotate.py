import os
import json
import mimetypes
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import google.generativeai as genai


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def get_api_key() -> str:
    """
    Resolve the Google Generative AI API key from environment, falling back to a hardcoded
    value present in the development notebook if not set. Prefer the environment for safety.
    """
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    # Fallback to the key observed in the notebook for local/dev convenience
    raise ValueError("GOOGLE_API_KEY is not set")


def analyze_meme_with_gemini(
    image_path: str,
    image_width: int,
    image_height: int,
    expected_indices: List[int],
    model: str = "gemini-2.5-pro",
    temperature: float = 0.4,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Run Gemini on a meme image with structured JSON output.

    Returns a dict with keys:
      - image_description: str
      - text_descriptions: [{index, position:{left,top,width,height}, description}]
    """
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    genai.configure(api_key=get_api_key())
    model_obj = genai.GenerativeModel(model)

    schema = {
        "type": "object",
        "properties": {
            "image_description": {"type": "string"},
            "text_descriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["index", "description"],
                },
            },
        },
        "required": ["image_description", "text_descriptions"],
    }

    prompt = f"""
    You are an expert meme analyst. Describe the provided meme image in detail.

    The image already contains red rectangular boxes drawn on top of it.
    Each box is labeled with a white numeric index (e.g., 0, 1, 2).

    For the image:
    - Explain what is happening, who or what is depicted, and any pop culture reference very briefly.
    - Describe the characters and how they’re commonly referred to.
    - Explain the situation and how this meme is typically used culturally.
    - Write the description as natural, expressive text (embedding-ready).
    - Share only what is relevant to the cultural and meme usage of the image.
    - Keep it concise (3-4 sentences)

    For each labeled text region:
    - Read the visible index number inside the red box and use it as "index".
    - Describe what kind of text typically goes there and why; mention spatial/contextual cues if helpful.
    - Explain the text's relevance to the meme
    - Ensure you return exactly one entry for each expected index provided.
    - Output all results in a single structured JSON following the schema.
    """

    user_data = {
        "image_size": {"width": image_width, "height": image_height},
        "expected_indices": expected_indices,
    }

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "temperature": temperature,
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model_obj.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime, "data": img_bytes}},
                            {"text": json.dumps(user_data)},
                        ],
                    }
                ],
                generation_config=generation_config,
            )
            return json.loads(resp.text)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                raise
            sleep_seconds = backoff_seconds * (2 ** (attempt - 1))
            logging.warning(
                "Gemini call failed on attempt %s/%s: %s; retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    # Should not be reached, but typing requires a return or raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown failure calling Gemini")


def chunk_iterable(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def build_image_path(repo_root: Path, filename: str) -> Path:
    annotated_dir = repo_root / "annotated_meme_images"
    return annotated_dir / filename


def process_single(
    repo_root: Path,
    meme_key: str,
    meme_data: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    image_path = build_image_path(repo_root, meme_data["filename"]).as_posix()

    result = analyze_meme_with_gemini(
        image_path=image_path,
        image_width=int(meme_data["width"]),
        image_height=int(meme_data["height"]),
        expected_indices=list(range(len(meme_data.get("text_options", [])))),
    )

    updated = dict(meme_data)
    updated["image_description"] = result.get("image_description", "")

    text_descs = result.get("text_descriptions", []) or []
    if "text_options" in updated and isinstance(updated["text_options"], list):
        for td in text_descs:
            try:
                idx = int(td.get("index", -1))
                if 0 <= idx < len(updated["text_options"]):
                    updated["text_options"][idx]["description"] = td.get("description", "")
            except Exception:  # noqa: BLE001
                # Skip malformed entries
                continue

    return meme_key, updated


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def main() -> None:
    configure_logging()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    memes_path = repo_root / "memes.json"
    if not memes_path.exists():
        raise FileNotFoundError(f"Could not find memes.json at {memes_path}")

    with memes_path.open("r", encoding="utf-8") as f:
        memes: Dict[str, Dict[str, Any]] = json.load(f)

    meme_items = list(memes.items())

    # Optionally skip those already annotated
    keys_to_process = [k for k, v in meme_items if v.get("image_description") is None]
    if not keys_to_process:
        logging.info("All memes already annotated. Nothing to do.")
        return
    
    logging.info(f"Annotating {len(keys_to_process)} memes")
    # Backup once at start
    backup_path = memes_path.with_suffix(".backup.json")
    if not backup_path.exists():
        logging.info("Creating backup at %s", backup_path)
        atomic_write_json(backup_path, memes)

    batch_size = 50
    max_workers = 50

    total = len(keys_to_process)
    processed = 0
    failures: List[str] = []

    for batch_keys in chunk_iterable(keys_to_process, batch_size):
        logging.info(
            "Processing batch of %d (progress %d/%d)",
            len(batch_keys),
            processed,
            total,
        )

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key in batch_keys:
                futures.append(
                    executor.submit(
                        process_single,
                        repo_root,
                        key,
                        memes[key],
                    )
                )

            for fut in as_completed(futures):
                try:
                    key, updated = fut.result()
                    memes[key] = updated
                    processed += 1
                except Exception as exc:  # noqa: BLE001
                    logging.error("Failed to process meme: %s", exc)
                    failures.append(str(exc))

        # Write after each batch
        logging.info("Writing updated JSON after batch. Processed so far: %d", processed)
        atomic_write_json(memes_path, memes)

    logging.info("Done. Total processed: %d. Failures: %d", processed, len(failures))


if __name__ == "__main__":
    main()


