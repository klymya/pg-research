#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMT="""# You are a **senior research engineer** specializing in computer vision and machine learning, with deep expertise in object detection, pose estimation, and model evaluation. You have extensive experience with YOLO-family architectures, COCO-style metrics, and production deployment of detection pipelines.

## YOUR TASK

You will receive an **HTML export of a Jupyter notebook** - a YOLOv8 evaluation report containing tables, plots, annotated images, and markdown commentary organized in Sections A-F.

You must:
1. Read and precisely analyze every element in the report (tables, plots, annotated images, text)
2. Produce a **JSON output** with two fields (see Output Format below)

---

## ANALYSIS METHOD

### Step 1 — Extract all data
- Read every table cell exactly. Do not round or approximate tabular values.
- For plots: identify axes, read trends, locate outliers. If a value cannot be read precisely from a chart, prefix with "~" (e.g., "~0.77").
- For annotated images: identify GT TP/Pred TP/GT FN/Pred FP boxes by color legend, note spatial patterns (edge vs. center, occluded vs. visible).

### Step 2 — Cross-reference across sections
- Link error composition (Section E) back to recall breakdowns (Section D).
- Verify that worst-frame metrics (Section C) are consistent with the F1 distribution (Section B) and visual summaries (Section F).
- Identify which specific attributes/conditions drive the global metric gaps (Section A).

### Step 3 — Identify root causes, not just symptoms
- "Recall is 0.93 for sitting regularly" is a symptom. The root cause might be annotation ambiguity, frame-edge truncation, or small apparent size at high camera heights. Determine which.

---

## OUTPUT FORMAT

Return a **single JSON object** with exactly two string fields. Each field contains **raw markdown** text that will be inserted directly into the Jupyter notebook as markdown cells.

```json
{
    "conclusions": "<markdown string>",
    "data_collection_requirements": "<markdown string>"
}
```

### Field 1: `conclusions`

**Audience:** R&D team and technical managers.
**Purpose:** Summarize main takeaways from Sections A–F. This will be appended to the notebook as Section G content.

**Style rules (match the existing notebook):**
- Use `## Section G - Conclusions & recommendations` as the top heading (already exists in notebook — your content goes under it)
- Use `### ` for sub-topics (e.g., `### Overall performance`, `### Error patterns`)
- Use `**bold**` for metric names and key values inline (e.g., **F1 = 0.975**)
- Use `- ` bullet lists for enumerating findings (same style as existing Section G draft)
- Use `| col | col |` markdown tables where a structured comparison helps
- Keep it **brief and dense** — no filler sentences. Every line should carry information.
- Reference section/table/figure numbers when citing evidence (e.g., "Table 3", "Figure 5")
- End with a short **Bottom line** statement (2-3 sentences max)

**Required content:**
1. Global performance summary (Section A numbers)
2. Frame-level distribution takeaway (Section B — what precentage is above/below thresholds)
3. Worst-case analysis (Section C — what went wrong and why)
4. Attribute-level recall patterns (Section D — which slices underperform)
5. Error composition insights (Section E — where do FNs come from)
6. Environmental factor effects (Section F — camera height, density, distance-from-center)
7. Bottom line: is the model ready, and what are the top risks?

### Field 2: `data_collection_requirements`

**Audience:** Data annotation team.
**Purpose:** Actionable guidance for planning the next data collection iteration. Will be appended as a markdown cell after the conclusions.

**Style rules (match the existing notebook):**
- Use `### Data collection and annotation requirements for the next iteration` as heading
- Start with 2-3 sentences of introduction explaining WHY these requirements were derived (link to specific findings)
- Then `#### Target scenario matrix (guidance for data collection)` as sub-heading
- Then a markdown table with exactly these columns: `| Case type | Description |`
- Case type = short label (e.g., "High-density meeting room", "Edge-truncated subjects")
- Description = 1-2 sentences: what to collect, how many, and why (tied to a specific finding)
- After the table: optionally add `#### Annotation guidelines` with bullet points if the analysis reveals annotation issues that need addressing

**Required content in the table:**
- Cover every significant failure mode or coverage gap found in the analysis
- Prioritize by impact (highest-impact cases first)
- Be specific: include camera heights, occupancy counts, scenario types, posture types where relevant
- Each row must trace back to evidence from the report

---

## QUALITY RULES

1. **Exact numbers only.** Never write "high recall" — write "recall = 0.967". Never write "most frames" — write "28/31 frames (90.3%)".

2. **Evidence-backed claims.** Every finding must cite a specific table, figure, or section. If a conclusion requires inference beyond what the data shows, mark it explicitly: *(inferred — not directly measured)*.

3. **Small sample awareness.** If a metric is based on fewer than 10 samples, note it. Example: "Recall = 1.0 for standing (n=X); small sample — interpret with caution."

4. **Notebook-native markdown.** The output will be pasted directly into Jupyter markdown cells. Do not use HTML tags, do not wrap in code fences, do not add cell metadata. Just clean markdown that renders correctly in Jupyter.

5. **Consistent voice.** Match the existing report's tone — technical, direct, no hedging. The existing notebook uses short declarative sentences and bulleted findings. Do the same.

6. **No repetition between fields.** The `conclusions` field analyzes what happened. The `data_collection_requirements` field says what to do next. Do not repeat the same content in both — reference conclusions from the requirements if needed ("Based on the edge-truncation errors identified above...").

---

## EXAMPLE OUTPUT STRUCTURE

```json
{
    "conclusions": "Global metrics show strong baseline: **box F1 = 0.975**, **pose F1 = 0.967** (Table 1).\n\n### Frame-level distribution\n- 28/31 frames (90.3%) achieve F1 > 0.9 (Table 2)\n- Only 1 frame below 0.8 ...\n\n### Error patterns\n- Dominant failure mode: ...\n...\n\n### Bottom line\n...",

    "data_collection_requirements": "### Data collection and annotation requirements for the next iteration\n\nAnalysis of Sections C–F reveals three coverage gaps ....\n\n#### Target scenario matrix (guidance for data collection)\n\n| Case type | Description |\n|-----------|-------------|\n| ... | ... |\n| ... | ... |\n\n#### Annotation guidelines\n- ..."
}
```

---

## CRITICAL REMINDERS

- You are writing Section G of a real engineering report. It must be publication-ready.
- The annotation team will use your table directly to plan field recordings. Be specific enough that they can act on it without asking follow-up questions.
- Technical managers will skim the conclusions. Lead with the most important finding, not background context.
- Output ONLY the JSON object. No preamble, no explanation, no markdown code fences around the JSON."""


NOTEBOOK_NAME = "extended_report.html"

# Regex for base64-encoded <img> tags (png and jpeg)
_BASE64_IMG_RE = re.compile(
    r'<img\s[^>]*?src="data:(image/(?:png|jpeg));base64,([^"]+)"[^>]*?>',
    re.IGNORECASE | re.DOTALL,
)
# Patterns for HTML boilerplate removal
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_EMPTY_DIV_RE = re.compile(r"<div[^>]*>\s*</div>", re.IGNORECASE)
# Heading text preceding an image (used for labelling)
_HEADING_RE = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)


def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None else default


def _load_env() -> None:
    env_path = os.getenv("ENV_PATH", ".env")
    load_dotenv(env_path)


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from a string, returning plain text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def preprocess_html(html_content: str) -> tuple[str, list[dict]]:
    """Extract base64 images and clean HTML boilerplate.

    Returns the cleaned HTML string and a list of image dicts with keys:
    ``media_type``, ``base64_data``, ``label``.
    """
    images: list[dict] = []
    heading_cache: list[str] = []

    # Collect all headings for context labelling
    for m in _HEADING_RE.finditer(html_content):
        heading_cache.append((m.start(), _strip_html_tags(m.group(1))))

    def _nearest_heading(pos: int) -> str:
        """Find the closest preceding heading to *pos*."""
        best = ""
        for h_pos, h_text in heading_cache:
            if h_pos <= pos:
                best = h_text
        return best

    def _replace_img(match: re.Match) -> str:
        media_type = match.group(1)
        base64_data = match.group(2)

        # Try to get alt text
        alt_match = re.search(r'alt="([^"]*)"', match.group(0), re.IGNORECASE)
        alt_text = alt_match.group(1).strip() if alt_match else ""

        idx = len(images) + 1
        heading_ctx = _nearest_heading(match.start())

        if alt_text:
            label = alt_text
        elif heading_ctx:
            label = f"Figure from: {heading_ctx}"
        else:
            label = f"Figure {idx}"

        images.append({
            "media_type": media_type,
            "base64_data": base64_data,
            "label": label,
        })
        return f"[IMAGE_{idx}: {label}]"

    # 1. Extract and replace base64 images
    cleaned = _BASE64_IMG_RE.sub(_replace_img, html_content)

    # 2. Strip boilerplate
    cleaned = _STYLE_RE.sub("", cleaned)
    cleaned = _SCRIPT_RE.sub("", cleaned)
    cleaned = _EMPTY_DIV_RE.sub("", cleaned)

    # 3. Collapse excessive whitespace (but keep single newlines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned, images


def estimate_tokens(text: str, chars_per_token: float = 3.5) -> int:
    """Rough token estimate based on character count."""
    return int(len(text) / chars_per_token)


def build_message(cleaned_html: str, images: list[dict]) -> HumanMessage:
    """Build multi-part HumanMessage with text + vision blocks."""
    content_blocks: list[dict] = [{"type": "text", "text": cleaned_html}]

    for img in images:
        content_blocks.append({
            "type": "image",
            "base64": img["base64_data"],
            "mime_type": img["media_type"],
        })
        content_blocks.append({
            "type": "text",
            "text": f"[Above image is: {img['label']}]",
        })

    return HumanMessage(content=content_blocks)


def analyze_report(notebook_path: str | Path, system_prompt: str) -> str:
    """Preprocess HTML, build message, call Claude, return result."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    cleaned_html, images = preprocess_html(raw_html)

    # --- Token-savings log ---
    raw_chars = len(raw_html)
    clean_chars = len(cleaned_html)
    raw_tokens = estimate_tokens(raw_html)
    clean_tokens = estimate_tokens(cleaned_html)
    saved = raw_tokens - clean_tokens
    pct = (saved / raw_tokens * 100) if raw_tokens else 0

    print(f"Original HTML: ~{raw_chars:,} chars (~{raw_tokens:,} estimated tokens)")
    print(f"Cleaned HTML:  ~{clean_chars:,} chars (~{clean_tokens:,} estimated tokens)")
    print(f"Images:        {len(images)} images (sent as vision tokens)")
    print(f"Estimated savings: ~{saved:,} tokens ({pct:.1f}%)")
    print()

    message = build_message(cleaned_html, images)
    model = ChatAnthropic(model="claude-opus-4-6", max_retries=3)
    response = model.invoke([SystemMessage(content=system_prompt), message])
    return response.text


def main():
    ap = argparse.ArgumentParser(
        description="Generate conclusions and data collection requirements "
        "from a YOLOv8 evaluation report using Anthropic's Claude model."
    )
    ap.add_argument(
        "--env-file",
        default="",
        help="(Optional) Path to .env file. If empty, uses ENV_PATH.",
    )

    args, _unknown = ap.parse_known_args()

    if args.env_file:
        load_dotenv(args.env_file)
    else:
        _load_env()

    project_path = _env("PROJECT_PTH", "")
    if project_path.startswith("/") and not project_path.startswith("/home/"):
        project_path = project_path[1:]
    notebook_path = Path.home() / project_path / NOTEBOOK_NAME

    result = analyze_report(notebook_path, SYSTEM_PROMT)
    print(result)


if __name__ == "__main__":
    raise SystemExit(main())
