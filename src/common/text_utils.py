"""Text processing utilities for base-36 Reddit IDs, content hashing, and string sanitization."""

import hashlib
import html
import re
from typing import Optional


def clean_reddit_id(raw_id: Optional[str]) -> str:
    """Safely cleans Reddit base-36 fullnames (e.g., 't3_1oxu8rh', 't1_abcdef') without destroying

    valid base-36 characters.

    Legacy defect: `.str.strip('t3_')` stripped any character in {'t', '3', '_'} from both ends.
    Fix: Use strict prefix replacement matching only leading '^t[13]_'.
    """
    if raw_id is None:
        return ""
    s = str(raw_id).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    # Strip leading Reddit fullname prefix: t1_ (comment), t3_ (submission/post)
    return re.sub(r"^t[13]_", "", s, flags=re.IGNORECASE)


def normalize_text(text: Optional[str]) -> str:
    """Normalizes raw Reddit text by decoding HTML entities, stripping zero-width spaces,

    and standardizing whitespace without truncating multiline paragraphs.
    """
    if text is None:
        return ""
    s = str(text)
    if not s or s.lower() in ("nan", "none", "null"):
        return ""

    # Decode HTML entities (&amp; -> &, &lt; -> <, etc.)
    s = html.unescape(s)

    # Remove null bytes and non-printable control characters (except newline and tab)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)

    # Standardize Windows / Unix line breaks
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse horizontal whitespace runs while keeping newlines intact
    s = re.sub(r"[ \t]+", " ", s)

    return s.strip()


def compute_content_hash(text: str) -> str:
    """Computes a deterministic SHA-256 hash of normalized alphanumeric text.

    Used for stage 2 deduplication to identify identical reposts regardless of
    minor casing, punctuation, or whitespace variations.
    """
    if not text:
        return ""
    # Lowercase and retain only alphanumeric characters and spaces
    simplified = re.sub(r"[^a-z0-9\s]", "", text.lower())
    collapsed = " ".join(simplified.split())
    if not collapsed:
        return ""
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()
