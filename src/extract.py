import types

from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
from config.settings import OPENROUTER_API_KEY

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

EXTRACTION_PROMPT = """
You are a FORMATTER and TRANSCRIBER of handwritten engineering notes.
Think of yourself as a typist — you type exactly what you see, formatted cleanly.

## YOUR TASK
Transcribe the handwritten content from this image into clean Markdown.
Improve readability ONLY — do not add, infer, or explain anything.

## OUTPUT FORMAT
- Headings    → # Main Heading / ## Subheading (only if writer underlined or made it clear)
- Bullets     → - item
- Inline math → $x[n]$
- Block math  → $$X(z) = \\sum_{n=-\\infty}^{\\infty} x[n] z^{-n}$$
- Tables      → standard Markdown table format
- Diagrams    → [DIAGRAM: describe only what is physically drawn, one sentence]
- Illegible   → [ILLEGIBLE]
- Margin note → ## Margin Notes section at the end

## ALLOWED (readability only)
✅ Wrapping math in LaTeX delimiters
✅ Converting underlined text to Markdown headings
✅ Fixing obvious typos ("teh" → "the")
✅ Formatting drawn bullet lists as Markdown bullets
✅ Keeping numbered equations with their number: $$...\\quad (1)$$

## NOT ALLOWED — STRICT
❌ Adding any sentence not written on the page
❌ Completing partial equations
❌ Adding "This means...", "Note:", "Therefore..." if not written
❌ Inferring diagram contents beyond what is physically drawn
❌ Writing math as plain text — always use LaTeX

## HALLUCINATION EXAMPLES — NEVER DO THIS
Page shows: "ROC excludes poles"
❌ BAD: "ROC excludes poles, meaning the system is stable when poles are inside the unit circle"
✅ GOOD: "ROC excludes poles"

Page shows: "$$H(z) = \\frac{1}{1 - az^{-1}}$$"
❌ BAD: "$$H(z) = \\frac{1}{1 - az^{-1}}$$ which represents a first order IIR filter"
✅ GOOD: "$$H(z) = \\frac{1}{1 - az^{-1}}$$"

## SELF-CHECK BEFORE RESPONDING
- Did I add any word or sentence not visible on the page? → remove it
- Is every equation in LaTeX? → fix it
- Did I complete anything the writer left incomplete? → undo it

## APPEND AT THE END (always)
---
FIDELITY_CHECK:
- added_content: none | minor | significant
- changed_equations: yes | no
- illegible_count: <number>
- confidence: high | medium | low
"""


def image_to_base64(img: Image.Image):
    # 🔥 Resize to reduce cost + improve speed
    img = img.resize((1024, 1024))

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_page(img: Image.Image, page_no: int) -> str:
    try:
        img_base64 = image_to_base64(img)

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=500,
        messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_base64}"
                        }
                    ]
                }
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error on page {page_no}: {e}")
        return ""

