import time
import base64
from io import BytesIO
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import warnings
from config.settings import GEMINI_API_KEY, GEMINI_MODEL

warnings.filterwarnings("ignore", category=UserWarning)

extraction_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    max_output_tokens=2048,
)

_RATE_LIMIT_DELAY = 5.5
_last_call_time = 0

EXTRACTION_PROMPT = """
You are a smart content extractor for engineering lecture slides and handwritten notes.
Your job is to extract ONLY the meaningful educational content — skip decorative elements.

## STEP 1: ASSESS THE PAGE
First, quickly assess what type of page this is:

A) **CONTENT PAGE** — Has definitions, explanations, equations, proofs, examples, 
   bullet points with substance, diagrams with labels, tables with data.
   → Extract everything meaningful.

B) **LOW-CONTENT PAGE** — Only has a title, lecture number, course name, date,
   instructor name, "Thank you", "Questions?", blank page, or purely decorative.
   Examples: "Lecture 9", "PCA", "End of Module 3", "Dr. Smith — CS301"
   → Respond with ONLY: [SKIP]

## STEP 2: IF CONTENT PAGE — EXTRACT AS FOLLOWS

### FORMAT RULES
- Headings    → # Main Heading / ## Subheading
- Bullets     → - item (keep the substance, skip redundant numbering like "Point 1:")
- Inline math → $x[n]$
- Block math  → $$X(z) = \\sum_{n=-\\infty}^{\\infty} x[n] z^{-n}$$
- Tables      → standard Markdown table format
- Diagrams    → [DIAGRAM: describe what is drawn — nodes, arrows, labels, axes]
- Code        → ```language ... ```
- Illegible   → [ILLEGIBLE]

### SMART EXTRACTION RULES
✅ Extract definitions, theorems, proofs, derivations
✅ Extract bullet points that contain actual information
✅ Extract equations and their context (what they represent)
✅ Extract examples and worked solutions
✅ Extract diagram descriptions if they convey information
✅ Extract table data
✅ Merge related fragments from the same concept into coherent text

❌ Skip slide numbers, page numbers, footer/header text
❌ Skip "Lecture N" or "Module N" if it's just a label (not a heading for content)
❌ Skip instructor name, course code, date, university name (unless part of content)
❌ Skip decorative borders, logos, watermarks
❌ Skip "Continued...", "cont'd", "See next slide"
❌ DO NOT add explanations, notes, or commentary not on the page
❌ DO NOT complete partial sentences or equations
❌ DO NOT add "This means...", "Note:", "In other words..." unless written on the page

### QUALITY RULES
- If a slide has BOTH a title AND content, keep the title as a heading and extract the content
- If a slide has only bullet fragments like "• Fast", "• Efficient", "• Scalable" — 
  extract them BUT include the slide title for context
- Preserve all mathematical notation in LaTeX format
- Keep numbered equations: $$...\\quad (1)$$

## SELF-CHECK BEFORE RESPONDING
- Is this a low-content page? → respond with [SKIP]
- Did I add any word not visible on the page? → remove it
- Is every equation in LaTeX? → fix it
- Did I skip the slide decorations? → good

## APPEND AT THE END (always, even for content pages)
---
FIDELITY_CHECK:
- added_content: none | minor | significant
- changed_equations: yes | no
- illegible_count: <number>
- confidence: high | medium | low
- content_density: high | medium | low | skip
"""


def image_to_base64(img: Image.Image) -> str:
    img = img.resize((1024, 1024))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def _rate_limit():
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)
    _last_call_time = time.time()


def _invoke_extraction(img_base64: str) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": EXTRACTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
            },
        ]
    )
    response = extraction_llm.invoke([message])
    content = response.content
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "\n".join(text_parts).strip()
    return str(content).strip()


def extract_page(img: Image.Image, page_no: int) -> str:
    try:
        _rate_limit()
        img_base64 = image_to_base64(img)
        result = _invoke_extraction(img_base64)

        if result.startswith("[SKIP]") or result == "[SKIP]":
            print(f"     ⏭️  Page {page_no} — skipped")
            return "[SKIP]"

        return result

    except Exception as e:
        print(f"❌ Error on page {page_no}: {e}")
        if "429" in str(e) or "quota" in str(e).lower():
            print("   ⏳ Rate limited, retrying in 60s...")
            time.sleep(60)
            try:
                global _last_call_time
                _last_call_time = time.time()
                img_base64 = image_to_base64(img)
                result = _invoke_extraction(img_base64)
                if result.startswith("[SKIP]"):
                    return "[SKIP]"
                return result
            except Exception as retry_err:
                print(f"❌ Retry failed on page {page_no}: {retry_err}")
                return ""
        return ""
