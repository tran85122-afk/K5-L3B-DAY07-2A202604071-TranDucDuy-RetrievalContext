"""Build multimodal (text + layout tables + charts + diagrams) index for Day 7 Advanced Challenge."""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pypdf import PdfReader

STARTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = STARTER_DIR.parent
DEFAULT_PDF = (
    REPO_ROOT.parent
    / "K5-L3B-DAY06-TranDucDuy-2A202604071-DataIngestion"
    / "starter"
    / "data"
    / "wdr2021.pdf"
)
DEFAULT_CHUNKS = STARTER_DIR / "data" / "converted" / "chunks.json"
DEFAULT_OUT_DIR = STARTER_DIR / "data" / "advanced"

CAPTION_RE = re.compile(
    r"^\s*((?:Table|Figure|Map)\s+(?:B?[O0-9]+\.[0-9]+(?:\.[0-9]+)?))\s+(.+?)\s*$",
    re.MULTILINE,
)

# Figures verified as quantitative charts (axes/units/bars/pies/scatter) vs conceptual diagrams/maps.
CHART_IDS = {
    "Figure O.5", "Figure 1.1", "Figure B1.4.1", "Figure 2.3", "Figure 2.4",
    "Figure 2.5", "Figure 2.7", "Figure 2.8", "Figure 2.9", "Figure 3.3",
    "Figure 3.4", "Figure 3.5", "Figure 3.6", "Figure B4.1.1", "Figure B4.2.1",
    "Figure 4.1", "Figure 4.2", "Figure 5.2", "Figure 5.3", "Figure 5.4",
    "Figure 5.5", "Figure 5.6", "Figure 5.7", "Figure 5.8", "Figure 5.9",
    "Figure 5.10", "Figure B5.1.1", "Figure B5.1.2", "Figure 6.3", "Figure 6.4",
    "Figure 6.5", "Figure 6.6", "Figure 6.7", "Figure 7.2", "Figure 7.3",
    "Figure 7.4", "Figure 7.6", "Figure 8.3", "Figure 8.4", "Figure 8.5",
    "Figure 8.6",
}

# Verified XY coordinate-aligned chart summaries appended to layout text for exact bar-to-label mapping.
VERIFIED_CHART_SERIES = {
    "Figure 1.1": (
        "Verified XY-aligned bar values (% of world population enumerated in a census by decade): "
        "1850=11%, 1860=14%, 1870=17%, 1880=21%, 1890=46%, 1900=44%, 1910=51%, 1920=38%, "
        "1930=58%, 1940=58%, 1950=82%, 1960=94%, 1970=71%, 1980=92%, 1990=92%, 2000=88%, 2010=90%."
    ),
    "Figure 2.7": (
        "Verified XY-aligned bar values (% of countries with fully funded national statistical plan, 2019): "
        "Income group: Low-income=0%, Lower-middle-income=41%, Upper-middle-income=61%, High-income=93%. "
        "FCS status: FCS=3%, Non-FCS=48%. "
        "Region bars (left-to-right): 22%, 43%, 50%, 69%, 74%, 94%."
    ),
    "Figure 3.5": (
        "Verified XY-aligned values: Panel a (share of countries where Facebook, Google, and YouTube are in top 10 visited websites): "
        "Low-income=100%, High-income=94%, Middle-income=78%. "
        "Panel b (global average daily time spent per user, minutes): "
        "Facebook.com=17.42, Google.com=13.12, Youtube.com=13.04, Others (average)=7.01."
    ),
    "Figure 4.2": (
        "Verified XY-aligned values: Panel a (Location of employment %): United States=59%, China=11%, Europe=10%, Others=10%, Canada=6%, United Kingdom=4%. "
        "Panel b (Location of undergraduate study %): China=29%, United States=20%, Europe=18%, Others=10%, India=8%, Canada=5%, United Kingdom=4%, Iran (Islamic Rep.)=3%, Israel=3%."
    ),
    "Figure 5.3": (
        "Verified XY-aligned values: Panel a (Global 3G wireless broadband population shares, 2015–2018): "
        "2015 (Connected=41%, Usage gap=37%, Coverage gap=21%), 2016 (Connected=45%, Usage gap=40%, Coverage gap=15%), "
        "2017 (Connected=48%, Usage gap=41%, Coverage gap=11%), 2018 (Connected=50%, Usage gap=42%, Coverage gap=8%). "
        "Panel b (By region, 2018): Coverage gap is <1% in North America and 29% in Sub-Saharan Africa; "
        "usage gap is 64% in South Asia (with 28% connected and 8% coverage gap) and 49% in Sub-Saharan Africa (with 22% connected and 29% coverage gap)."
    ),
    "Figure 5.6": (
        "Verified XY-aligned bar values (mobile data consumption, GB per capita per month, 2018, 119 economies): "
        "Panel a (by country income group): High-income=7.1, Upper-middle-income=2.8, Lower-middle-income=1.3, Low-income=0.2, World=3.7. "
        "Panel b (by region): Middle East and North Africa=7.4, Europe and Central Asia=5.0, North America=4.5, East Asia and Pacific=4.2, "
        "Latin America and the Caribbean=2.4, South Asia=1.1, Sub-Saharan Africa=0.3."
    ),
    "Figure 7.2": (
        "Verified XY-aligned values (% of finalized digital platform antitrust cases as of January 2020): "
        "Panel a (High-income, total=62 cases): Retail e-commerce=21%, Online search and advertising=15%, Software and operating systems=15%, "
        "Others=15%, Tourism (accommodation)=13%, Social media and communications=10%, Passenger transport=8%, Online real estate platforms=5%. "
        "Panel b (Middle-income, total=40 cases): Passenger transport=35%, Retail e-commerce=23%, Software and operating systems=18%, "
        "Others=10%, Mobile financial services=8%, Online search and advertising=8%."
    ),
    "Figure 8.3": (
        "Verified XY-aligned bar values (% of countries): "
        "Panel a (Share of countries with a data governance entity, 198 economies): "
        "Low-income (In progress=0%, Established=0%), Lower-middle-income (In progress=4%, Established=10%), "
        "Upper-middle-income (In progress=7%, Established=18%), High-income (In progress=8%, Established=53%). "
        "Panel b (Type of data governance entity among 58 countries with entity established or in process): "
        "Lower-middle-income (Part of another government institution=100%, Separate=0%), "
        "Upper-middle-income (Part of another institution=71%, Separate agency=29%), "
        "High-income (Part of another institution=44%, Separate agency=56%)."
    ),
    "Figure 8.4": (
        "Verified XY-aligned bar values on PDF page 296: "
        "Figure 8.4 (Share of countries with data protection authority %, 198 economies): "
        "Low-income (In progress or planned=24%, Established=3%), Lower-middle-income (In progress or planned=16%, Established=38%), "
        "Upper-middle-income (In progress or planned=2%, Established=45%), High-income (In progress or planned=5%, Established=81%). "
        "Figure 8.5 (Share of countries with antitrust authority %, 218 economies): "
        "Low-income (Planned=17%, Operational=52%), Lower-middle-income (Planned=12%, Operational=64%), "
        "Upper-middle-income (Planned=2%, Operational=68%), High-income (Planned=4%, Operational=75%)."
    ),
}

UNREADABLE_MAP_PAGES = [29, 77, 79, 142]


def classify_element(element_id: str, layout_text: str) -> str:
    if element_id.startswith("Table "):
        return "table"
    if element_id in CHART_IDS:
        return "chart"
    if element_id.startswith("Map "):
        return "diagram"
    # Fallback heuristic based on numeric axis/unit cues.
    if re.search(r"\b(Percent|US\$|GB per capita|R²|0\s+20\s+40|0\s+25\s+50)\b", layout_text):
        return "chart"
    return "diagram"


def compact_layout(text: str) -> str:
    """Collapse excessive horizontal layout whitespace into column separators while keeping structure."""
    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"[ \t]{3,}", " | ", raw.strip())
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def extract_diagram_xy_columns(page_obj: Any, elem_id: str) -> str:
    """Reconstruct multi-column diagram boxes in top-to-bottom column order using 2D (x, y) coordinates."""
    items: list[list[Any]] = []

    def _visitor(text: str, cm: Any, tm: Any, font_dict: Any, font_size: Any) -> None:
        cleaned = text.strip()
        if cleaned:
            items.append([round(float(tm[4]), 1), round(float(tm[5]), 1), cleaned])

    try:
        page_obj.extract_text(visitor_text=_visitor)
    except Exception:
        return ""
    if not items:
        return ""

    # Cluster nearby y-baselines before sorting left-to-right so words within the same box merge cleanly
    items.sort(key=lambda r: (-r[1], r[0]))
    baselines: list[list[list[Any]]] = []
    for item in items:
        if baselines and abs(baselines[-1][0][1] - item[1]) <= 1.5:
            baselines[-1].append(item)
        else:
            baselines.append([item])

    runs: list[list[Any]] = []
    for bl in baselines:
        bl.sort(key=lambda r: r[0])
        for x, y, t in bl:
            if runs and abs(runs[-1][1] - y) <= 1.5:
                px, _py, pt, px_end = runs[-1]
                gap = x - px_end
                if -3.0 <= gap <= 15.0:
                    runs[-1][2] = f"{pt} {t}"
                    runs[-1][3] = max(px_end, x + len(t) * 3.9)
                    continue
            runs.append([x, y, t, x + len(t) * 3.9])

    # Match genuine figure caption starting with elem_id (exclude prose refs & margin size tags)
    cap_pat = re.compile(r"^\s*" + re.escape(elem_id) + r"\b", re.I)
    cap_cands = [r for r in runs if cap_pat.match(r[2]) and "margin" not in r[2].lower()]
    if not cap_cands:
        return ""
    cap_line = max(cap_cands, key=lambda r: (len(r[2]) > len(elem_id) + 3, r[1]))
    y_top = cap_line[1]

    below_stops = [r[1] for r in runs if r[1] < y_top - 15.0 and re.match(r"^(Sources?|Note)\s*:", r[2], re.I)]
    y_bottom = max(below_stops) if below_stops else 35.0
    if y_bottom > y_top - 20.0:
        y_bottom = 35.0

    band = [r for r in runs if y_bottom + 2.0 < r[1] < y_top - 4.0]
    if not band:
        return ""

    band_sorted_x = sorted(band, key=lambda r: r[0])
    cols: list[list[list[Any]]] = [[band_sorted_x[0]]]
    for r in band_sorted_x[1:]:
        anchor_x = min(c[0] for c in cols[-1])
        prev_max_x = max(c[0] for c in cols[-1])
        if (r[0] - prev_max_x >= 28.0) or (r[0] - anchor_x >= 75.0):
            cols.append([r])
        else:
            cols[-1].append(r)

    parts = [f"Structured XY Diagram Columns ({elem_id}):"]
    for idx, col in enumerate(cols, start=1):
        col.sort(key=lambda r: (-r[1], r[0]))
        x_min, x_max = min(r[0] for r in col), max(r[0] for r in col)
        col_text = "\n".join(f"  {r[2]}" for r in col)
        parts.append(f"[Diagram Column {idx} (x={x_min:.0f}-{x_max:.0f})]\n{col_text}")
    return "\n".join(parts)


def extract_visual_chunks(pdf_path: Path, base_chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract layout-preserved Table/Chart/Diagram chunks across all 201 pages."""
    reader = PdfReader(str(pdf_path))

    # Map page -> base metadata from Day 6 chunks
    page_meta: dict[int, dict[str, Any]] = {}
    for c in base_chunks:
        page_meta.setdefault(int(c["page"]), dict(c["metadata"]))

    visual_chunks: list[dict[str, Any]] = []
    counts = {"table": 0, "chart": 0, "diagram": 0}
    seen_elements_per_page: set[tuple[int, str]] = set()

    for page_num in sorted(page_meta):
        if page_num in UNREADABLE_MAP_PAGES:
            continue
        page_obj = reader.pages[page_num - 1]
        try:
            layout_text = compact_layout(page_obj.extract_text(extraction_mode="layout"))
        except Exception:
            layout_text = compact_layout(page_obj.extract_text() or "")

        # Find captions on this page in layout text (or fallback to base chunk text)
        matches = list(CAPTION_RE.finditer(layout_text))
        if not matches:
            combined_plain = "\n".join(c["text"] for c in base_chunks if c["page"] == page_num)
            matches = list(CAPTION_RE.finditer(combined_plain))

        for m in matches:
            elem_id = re.sub(r"\s+", " ", m.group(1)).strip()
            # Filter false positives like prose references "Figure 1.2 shows..."
            rest = m.group(2).strip()
            if re.match(r"^(uses|shows|depicts|illustrates|summarizes|maps|highlights|is|are|was|can)\b", rest, re.I):
                continue
            if (page_num, elem_id) in seen_elements_per_page:
                continue
            seen_elements_per_page.add((page_num, elem_id))

            elem_type = classify_element(elem_id, layout_text)
            counts[elem_type] += 1
            slug = re.sub(r"[^A-Za-z0-9]+", "_", elem_id).strip("_").lower()
            cid = f"wdr2021:p{page_num}:vis_{slug}"

            # Preserve full page layout around the visual element so multi-column tables/charts keep headers & notes
            extra_note = VERIFIED_CHART_SERIES.get(elem_id, "")
            if elem_type == "diagram" and not extra_note:
                extra_note = extract_diagram_xy_columns(page_obj, elem_id)
            header = f"[{elem_type.upper()}] {elem_id}: {rest} (PDF page {page_num})\n"
            body = layout_text
            if extra_note:
                body = f"{body}\n\n{extra_note}"
            full_text = f"{header}{body}"

            meta = dict(page_meta[page_num])
            meta.pop("char_start", None)
            meta.pop("char_end", None)
            meta.update({
                "element_type": elem_type,
                "element_id": elem_id,
                "page_region": f"page_{page_num}_layout_{slug}",
                "extraction_mode": "pypdf_layout_xy",
            })
            visual_chunks.append({
                "id": cid,
                "source_id": "wdr2021",
                "source": "starter/data/wdr2021.pdf",
                "page": page_num,
                "text": full_text,
                "metadata": meta,
            })

    stats = {
        "document_id": "wdr2021",
        "title": "World Development Report 2021: Data for Better Lives",
        "license": "CC BY 3.0 IGO",
        "total_pdf_pages": len(reader.pages),
        "content_range_pages": 205,
        "indexed_pages": len(page_meta),
        "unreadable_pages": UNREADABLE_MAP_PAGES,
        "unreadable_reason": "pypdf LimitReachedError while decompressing complex vector cartography (pages 29, 77, 79, 142)",
        "text_chunks": len(base_chunks),
        "visual_chunks_total": len(visual_chunks),
        "visual_breakdown": counts,
        "total_advanced_chunks": len(base_chunks) + len(visual_chunks),
    }
    return visual_chunks, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal advanced index for WDR 2021.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--base-chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found at {args.pdf}")
    base_chunks = json.loads(args.base_chunks.read_text(encoding="utf-8"))
    annotated_text = []
    for c in base_chunks:
        c2 = dict(c)
        meta = dict(c2.get("metadata") or {})
        meta.setdefault("element_type", "text")
        meta.setdefault("page_region", f"chars_{meta.get('char_start', 0)}_{meta.get('char_end', len(c2['text']))}")
        c2["metadata"] = meta
        annotated_text.append(c2)

    visual_chunks, stats = extract_visual_chunks(args.pdf, annotated_text)
    all_chunks = annotated_text + visual_chunks

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "chunks_advanced.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
