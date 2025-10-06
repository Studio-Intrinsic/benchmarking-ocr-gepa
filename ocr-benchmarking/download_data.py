"""
Dataset Downloader for OCR Benchmarking

Downloads the OCR benchmark dataset from HuggingFace and caches images locally.
The dataset comes from getomni-ai/ocr-benchmark and includes:
  - Document images (PDFs, forms, receipts, etc.)
  - Ground truth markdown for each document
  - Ground truth JSON with structured data
  - JSON schemas for extraction tasks

Usage:
    python download_data.py                    # Download 1000 documents with images
    python download_data.py --no-images        # Download JSON only, skip images
    python download_data.py --max-rows 100     # Download only 100 documents
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from io import BytesIO
from PIL import Image


########################################
# IMAGE CACHING UTILITIES
########################################

def get_image_hash(image_url: str) -> str:
    """Generate MD5 hash for image URL (matches TypeScript implementation)."""
    return hashlib.md5(image_url.encode()).hexdigest()


def get_image_extension(image_url: str, content_type: str = None) -> str:
    """Extract image extension from URL or content type."""
    # Try URL first
    parsed = urllib.parse.urlparse(image_url)
    path_ext = Path(parsed.path).suffix.lower()

    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    if path_ext in valid_extensions:
        return path_ext

    # Fallback to content-type
    if content_type:
        type_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }
        return type_map.get(content_type.lower(), ".jpg")

    return ".jpg"  # Default fallback


def download_image_to_cache(
    image_url: str, cache_dir: Path, doc_idx: int = None
) -> tuple[bool, bool]:
    """Download image and save as PNG in cache directory.

    Always converts images to PNG format for consistency. If an image already
    exists in a different format (jpg, webp, etc.), converts it to PNG.

    Args:
        image_url: URL of the image to download
        cache_dir: Directory to save the image
        doc_idx: Document index (used for filename)

    Returns:
        Tuple of (success, was_already_cached)
    """
    if not image_url or not image_url.startswith(("http://", "https://")):
        return False, False

    try:
        # If we have a document index, prefer the exact PNG filename
        if doc_idx is not None:
            png_path = cache_dir / f"{doc_idx}.png"
            if png_path.exists():
                print(f"  Doc {doc_idx}: Image already cached at {png_path.name}")
                return True, True

            # Backward-compatibility: if a non-PNG exists, convert it to PNG once
            for ext in [".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
                legacy_path = cache_dir / f"{doc_idx}{ext}"
                if legacy_path.exists():
                    try:
                        with Image.open(legacy_path) as im:
                            im = (
                                im.convert("RGBA")
                                if im.mode in ("P", "RGBA", "LA")
                                else im.convert("RGB")
                            )
                            tmp_png = cache_dir / f"{doc_idx}.tmp.png"
                            im.save(tmp_png, format="PNG")
                            tmp_png.replace(png_path)
                        print(
                            f"  Doc {doc_idx}: Converted existing {legacy_path.name} -> {png_path.name}"
                        )
                        return True, False
                    except Exception:
                        # If conversion fails, proceed to re-download
                        pass

        # Download the image
        if doc_idx is not None:
            print(f"  Doc {doc_idx}: Downloading image...")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DocumentParser/1.0)"}
        resp = requests.get(image_url, headers=headers, timeout=30, stream=True)
        resp.raise_for_status()

        # Read full content into memory for conversion
        buf = BytesIO()
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                buf.write(chunk)
        buf.seek(0)

        # Open and convert to PNG
        with Image.open(buf) as im:
            im = (
                im.convert("RGBA")
                if im.mode in ("P", "RGBA", "LA")
                else im.convert("RGB")
            )

            if doc_idx is not None:
                final_path = cache_dir / f"{doc_idx}.png"
                temp_path = cache_dir / f"{doc_idx}.tmp.png"
            else:
                hash_name = get_image_hash(image_url)
                final_path = cache_dir / f"{hash_name}.png"
                temp_path = cache_dir / f"{hash_name}.tmp.png"

            im.save(temp_path, format="PNG")
            temp_path.replace(final_path)

        if doc_idx is not None:
            print(f"  Doc {doc_idx}: Downloaded: {final_path.name}")

        return True, False

    except Exception as e:
        if doc_idx is not None:
            print(f"  Doc {doc_idx}: Failed to cache image: {e}")
        # Cleanup temp file if it exists (both index and hash variants)
        try:
            if doc_idx is not None:
                temp_path = cache_dir / f"{doc_idx}.tmp.png"
                if temp_path.exists():
                    temp_path.unlink()
            else:
                temp_path = cache_dir / f"{get_image_hash(image_url)}.tmp.png"
                if temp_path.exists():
                    temp_path.unlink()
        except Exception:
            pass
        return False, False


def get_image_for_json(
    json_index: int, cache_dir: Optional[Path] = None
) -> Optional[Path]:
    """Get the local image file path for a given JSON index.

    Args:
        json_index: The JSON file index (e.g., 0 for 0.json)
        cache_dir: Image cache directory (defaults to ./image-cache)

    Returns:
        Path to the image file, or None if not found

    Example:
        # Get image for 0.json
        img_path = get_image_for_json(0)
        if img_path:
            print(f"Image for 0.json: {img_path}")
    """
    if cache_dir is None:
        cache_dir = Path("image-cache")

    # Prefer the standardized PNG filename
    png_path = cache_dir / f"{json_index}.png"
    if png_path.exists():
        return png_path

    # Backward-compatibility: check other extensions
    for ext in [".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
        legacy_path = cache_dir / f"{json_index}{ext}"
        if legacy_path.exists():
            return legacy_path

    return None


def get_json_for_image(image_path: Path, data_dir: Path = None) -> Optional[Path]:
    """Get the JSON file path for a given image path.

    Args:
        image_path: Path to the image file (either index-based or hash-based)
        data_dir: Data directory containing JSON files (defaults to ./data)

    Returns:
        Path to the JSON file, or None if not found

    Example:
        # From index-based image name
        json_path = get_json_for_image(Path("image-cache/0.jpg"))
        # Returns: Path("data/0.json")

        # From hash-based image name (finds all JSON files that reference it)
        json_path = get_json_for_image(Path("image-cache/d58b6eb91748e99b2a5098bb1b9a7870.jpg"))
    """
    if data_dir is None:
        data_dir = Path("data")

    image_name = image_path.name

    # Case 1: Index-based naming (0.jpg -> 0.json)
    if image_name.split(".")[0].isdigit():
        json_index = int(image_name.split(".")[0])
        json_path = data_dir / f"{json_index}.json"
        if json_path.exists():
            return json_path

    # Case 2: Hash-based naming - need to find which JSON files reference this image
    # This is more expensive but handles the case where you only have the hash filename
    cache_dir = image_path.parent

    # Look for any index-based symlinks pointing to this image
    for potential_link in cache_dir.glob("*.jpg"):
        if (
            potential_link.is_symlink()
            and potential_link.resolve() == image_path.resolve()
        ):
            link_name = potential_link.name
            if link_name.split(".")[0].isdigit():
                json_index = int(link_name.split(".")[0])
                json_path = data_dir / f"{json_index}.json"
                if json_path.exists():
                    return json_path

    # Also check other extensions
    for ext in [".png", ".gif", ".webp", ".bmp"]:
        for potential_link in cache_dir.glob(f"*{ext}"):
            if (
                potential_link.is_symlink()
                and potential_link.resolve() == image_path.resolve()
            ):
                link_name = potential_link.name
                if link_name.split(".")[0].isdigit():
                    json_index = int(link_name.split(".")[0])
                    json_path = data_dir / f"{json_index}.json"
                    if json_path.exists():
                        return json_path

    return None


def get_document_pair(
    index: int, data_dir: Path = None, cache_dir: Path = None
) -> tuple[Optional[Path], Optional[Path]]:
    """Get both JSON and image paths for a document index.

    Args:
        index: Document index (0-999)
        data_dir: Directory containing JSON files (defaults to ./data)
        cache_dir: Directory containing images (defaults to ./image-cache)

    Returns:
        Tuple of (json_path, image_path), either can be None if not found

    Example:
        json_path, img_path = get_document_pair(0)
        if json_path and img_path:
            print(f"Document 0: {json_path} + {img_path}")
    """
    if data_dir is None:
        data_dir = Path("data")
    if cache_dir is None:
        cache_dir = Path("image-cache")

    # Get JSON path
    json_path = data_dir / f"{index}.json"
    json_path = json_path if json_path.exists() else None

    # Get image path
    img_path = get_image_for_json(index, cache_dir)

    return json_path, img_path


def list_all_document_pairs(
    data_dir: Path = None, cache_dir: Path = None
) -> list[tuple[int, Optional[Path], Optional[Path]]]:
    """List all available document pairs with their indices.

    Args:
        data_dir: Directory containing JSON files (defaults to ./data)
        cache_dir: Directory containing images (defaults to ./image-cache)

    Returns:
        List of (index, json_path, image_path) tuples

    Example:
        pairs = list_all_document_pairs()
        for idx, json_path, img_path in pairs:
            if json_path and img_path:
                print(f"Complete pair {idx}: {json_path.name} + {img_path.name}")
            else:
                print(f"Incomplete pair {idx}: JSON={json_path is not None}, IMG={img_path is not None}")
    """
    if data_dir is None:
        data_dir = Path("data")
    if cache_dir is None:
        cache_dir = Path("image-cache")

    pairs = []

    # Find all JSON files and check for corresponding images
    if data_dir.exists():
        for json_file in sorted(data_dir.glob("*.json")):
            try:
                index = int(json_file.stem)
                img_path = get_image_for_json(index, cache_dir)
                pairs.append((index, json_file, img_path))
            except ValueError:
                # Skip non-numeric JSON files
                continue

    return pairs


def download_images_batch(
    image_urls_with_indices: list[tuple[str, int]],
    cache_dir: Path,
    max_workers: int = 10,
) -> dict:
    """Download multiple images concurrently. Returns dict with stats."""
    if not image_urls_with_indices:
        return {"cached": 0, "downloaded": 0, "failed": 0}

    downloaded_count = 0
    cached_count = 0
    failed_count = 0

    print(
        f"  Downloading {len(image_urls_with_indices)} images with {max_workers} concurrent workers..."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_idx = {}
        for image_url, doc_idx in image_urls_with_indices:
            if image_url:  # Only submit if URL exists
                future = executor.submit(
                    download_image_to_cache, image_url, cache_dir, doc_idx
                )
                future_to_idx[future] = doc_idx

        # Process completed downloads
        for future in as_completed(future_to_idx):
            try:
                success, was_cached = future.result()
                if success:
                    if was_cached:
                        cached_count += 1
                    else:
                        downloaded_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                doc_idx = future_to_idx[future]
                print(f"  Doc {doc_idx}: Unexpected error: {e}")
                failed_count += 1

    return {
        "cached": cached_count,
        "downloaded": downloaded_count,
        "failed": failed_count,
    }


########################################
# UTILITY HELPERS
########################################

def maybe_parse_json(value: Any) -> Any:
    """Parse a JSON-encoded string if possible."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def build_metadata(
    src_meta: Dict | None, width: int | None, height: int | None
) -> Dict:
    """Create the required metadata block with sane defaults."""
    src_meta = src_meta or {}
    meta = {
        "orientation": src_meta.get("orientation", 0),
        "documentQuality": str(src_meta.get("documentQuality", "")).lower()
        or "unknown",
        "language": src_meta.get("language", "EN"),
    }
    if width is not None and height is not None:
        meta["resolution"] = [width, height]
    return meta


########################################
# SCHEMA SANITIZER
########################################

def sanitise_schema(node: Any) -> None:
    """Recursively fix JSON schema issues for vision model compatibility.

    Fixes common schema problems:
      - Invalid 'type': 'enum' (should be string or number)
      - Typo 'emum' → 'enum'
      - Unsupported date formats
    """
    # Lists: recurse into each element
    if isinstance(node, list):
        for item in node:
            sanitise_schema(item)
        return

    # Non-dict primitives: nothing to do
    if not isinstance(node, dict):
        return

    # 1️⃣  Repair invalid 'type': 'enum'
    if node.get("type") == "enum":
        enum_vals = node.get("enum", [])
        node["type"] = (
            "number"
            if all(isinstance(v, (int, float)) for v in enum_vals)
            else "string"
        )

    # 2️⃣  Fix misspelled 'emum'
    if "emum" in node:
        node["enum"] = node.pop("emum")

    # 3️⃣  Normalise unsupported string formats
    if node.get("type") == "string":
        fmt = node.get("format")
        if fmt == "date":
            node["format"] = "date-time"
        elif fmt not in (None, "date-time"):
            node.pop("format", None)

    # Recurse into *every* value to be safe
    for val in node.values():
        sanitise_schema(val)


########################################
# ROW TRANSFORMER
########################################

def transform_row(raw_row: Dict) -> Dict:
    """Convert a HuggingFace dataset row to our JSON format.

    Extracts image URL, metadata, JSON schema, and ground truth outputs.
    """
    # Image block → imageUrl
    if isinstance(raw_row.get("image"), dict):
        image_url = raw_row["image"].get("src")
        width = raw_row["image"].get("width")
        height = raw_row["image"].get("height")
    else:
        image_url = raw_row.get("image") or raw_row.get("imageUrl")
        width = height = None

    # Metadata
    metadata = build_metadata(raw_row.get("metadata"), width, height)

    # Clean the schema in-place
    schema = raw_row.get("json_schema")
    if schema is not None:
        sanitise_schema(schema)

    return {
        "imageUrl": image_url,
        "metadata": metadata,
        "jsonSchema": schema,
        "trueJsonOutput": raw_row.get("true_json_output"),
        "trueMarkdownOutput": raw_row.get("true_markdown_output"),
    }


########################################
# MAIN DOWNLOAD LOOP
########################################

def main() -> None:
    """Download OCR benchmark dataset from HuggingFace.

    Downloads JSON files to ./data/ and caches images to ./image-cache/
    Fetches from getomni-ai/ocr-benchmark dataset in batches.
    """
    parser = argparse.ArgumentParser(
        description="Download OCR benchmark data and optionally cache images"
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Skip downloading images to cache"
    )
    parser.add_argument(
        "--max-rows", type=int, default=1000, help="Maximum number of rows to download"
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=20,
        help="Number of concurrent image download workers",
    )
    args = parser.parse_args()

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # Create image cache directory (matches TypeScript implementation)
    cache_dir = Path("image-cache")
    if not args.no_images:
        cache_dir.mkdir(exist_ok=True)
        print(f"Image cache directory: {cache_dir.absolute()}")
        print(f"Image download workers: {args.image_workers}")
    else:
        print("Skipping image caching (--no-images specified)")

    max_rows, batch_size = args.max_rows, 100
    total_downloaded = 0
    total_cached = 0
    total_failed = 0

    for offset in range(0, max_rows, batch_size):
        print(f"\nFetching rows {offset}–{offset + batch_size - 1}")

        resp = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": "getomni-ai/ocr-benchmark",
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": batch_size,
            },
            timeout=30,
        )
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["rows"])

        # Collect image URLs for concurrent download
        image_urls_batch = []

        for i, row in df.iterrows():
            idx = offset + i
            raw_row = {k: maybe_parse_json(v) for k, v in row["row"].items()}
            clean_row = transform_row(raw_row)

            # Save JSON file
            with open(out_dir / f"{idx}.json", "w") as f:
                json.dump(clean_row, f, indent=2)

            # Collect image URL for batch download
            if not args.no_images:
                image_url = clean_row.get("imageUrl")
                if image_url:
                    image_urls_batch.append((image_url, idx))

        # Download all images in this batch concurrently
        if not args.no_images and image_urls_batch:
            print(f"Processing {len(image_urls_batch)} images concurrently...")
            batch_stats = download_images_batch(
                image_urls_batch, cache_dir, args.image_workers
            )
            total_cached += batch_stats["cached"]
            total_downloaded += batch_stats["downloaded"]
            total_failed += batch_stats["failed"]

            print(
                f"Batch complete: {batch_stats['cached']} cached, {batch_stats['downloaded']} downloaded, {batch_stats['failed']} failed"
            )

    if not args.no_images:
        print(f"\nImage caching summary:")
        print(f"  Already cached: {total_cached}")
        print(f"  Newly downloaded: {total_downloaded}")
        print(f"  Failed downloads: {total_failed}")
        print(f"  Total processed: {total_cached + total_downloaded + total_failed}")

        # Show final cache stats (PNG-only expected)
        cache_files = list(cache_dir.glob("*.png"))
        total_size = sum(f.stat().st_size for f in cache_files)
        print(
            f"  Cache size (PNG): {len(cache_files)} files, {total_size / 1024 / 1024:.1f}MB"
        )

    print(
        f"\nDownload complete! Downloaded {max_rows} documents to {out_dir.absolute()}"
    )


if __name__ == "__main__":
    # no images
    # python3 download_data.py --no-images --max-rows 1000
    main()
