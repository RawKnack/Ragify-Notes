import re


def clean_text(text: str) -> str:
    text = text.replace("```", "").strip()
    for label in ("plaintext", "markdown", "text"):
        if text.lower().startswith(label):
            text = text[len(label):].strip()
            break
    return text


def extract_title(text: str) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        if line.startswith(("*", "-", "->", "•")):
            continue
        return line
    return "General Notes"


def extract_headings(text: str) -> list[str]:
    headings = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            headings.append(line.lstrip("#").strip())
    return headings


def parse_fidelity(text: str) -> dict:
    fidelity = {
        "added_content": "unknown",
        "changed_equations": "unknown",
        "illegible_count": 0,
        "confidence": "medium",
    }

    patterns = {
        "added_content":     r"added_content:\s*(\w+)",
        "changed_equations": r"changed_equations:\s*(\w+)",
        "illegible_count":   r"illegible_count:\s*(\d+)",
        "confidence":        r"confidence:\s*(\w+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1)
            fidelity[key] = int(val) if key == "illegible_count" else val

    return fidelity


def clean_fidelity_block(text: str) -> str:
    # Split on the fidelity block and keep only the transcription above it
    return re.split(r"---\s*FIDELITY_CHECK:", text)[0].strip()


def structure_page(text: str, page_no: int) -> dict:
    # Parse fidelity BEFORE cleaning (block lives at bottom of raw text)
    fidelity = parse_fidelity(text)

    # Remove fidelity block, then clean text
    text = clean_fidelity_block(text)
    text = clean_text(text)

    # Warn in terminal for suspicious pages
    if fidelity["added_content"] in ("minor", "significant"):
        print(f"⚠️  Page {page_no} flagged — model reports added content")
    if fidelity["changed_equations"] == "yes":
        print(f"⚠️  Page {page_no} flagged — model reports changed equations")

    return {
        "id": str(page_no),
        "text": text,
        "metadata": {
            "page_no": str(page_no),
            "section": extract_title(text),
            "headings": extract_headings(text),
            "word_count": len(text.split()),
            "fidelity": fidelity,
            "confidence": fidelity["confidence"],
        },
    }