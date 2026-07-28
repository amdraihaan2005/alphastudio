import fitz
import logging
from typing import Generator, TypedDict

logger = logging.getLogger(__name__)


class ParsedPage(TypedDict):
    page_number: int
    text: str


def table_to_markdown(table_data: list[list[str | None]]) -> str:
    """
    Converts raw table list of lists from PyMuPDF into a formatted Markdown table.
    """
    if not table_data or not table_data[0]:
        return ""

    headers = [
        str(cell or "").strip().replace("\n", " ").replace("|", "\\|")
        for cell in table_data[0]
    ]
    if not any(headers):
        headers = [f"Col {i + 1}" for i in range(len(table_data[0]))]

    rows = []
    for r in table_data[1:]:
        clean_row = [
            str(cell or "").strip().replace("\n", " ").replace("|", "\\|") for cell in r
        ]
        rows.append(clean_row)

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = []
    for row in rows:
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[: len(headers)]
        row_lines.append("| " + " | ".join(row) + " |")

    return (
        "\n" + header_line + "\n" + separator_line + "\n" + "\n".join(row_lines) + "\n"
    )


def is_overlapping(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    threshold: float = 0.4,
) -> bool:
    """
    Checks if bounding box A overlaps bounding box B by more than the threshold ratio of A's area.
    Bounding box format: (x0, y0, x1, y1)
    """
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return False

    intersection_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)

    if area_a <= 0:
        return False

    return (intersection_area / area_a) >= threshold


def parse_pdf_pages(file_path: str) -> Generator[ParsedPage, None, None]:
    """
    Lazily streams parsed page text and tables from a PDF file.
    Keeps RAM usage extremely low by only loading one page at a time.
    """
    logger.info(f"Opening PDF file for streaming parsing: {file_path}")
    doc = None
    try:
        doc = fitz.open(file_path)
        for page_idx, page in enumerate(doc):
            page_number = page_idx + 1

            tables = page.find_tables()
            table_bboxes = []
            table_markdowns = {}

            for t in tables.tables:
                bbox = t.bbox
                table_bboxes.append(bbox)
                try:
                    table_md = table_to_markdown(t.extract())
                    table_markdowns[bbox] = table_md
                except Exception as table_err:
                    logger.warning(
                        f"Failed to extract table on page {page_number}: {table_err}"
                    )

            blocks = page.get_text("blocks")
            elements = []

            for bbox, md in table_markdowns.items():
                elements.append((bbox[1], bbox[0], "table", md, bbox))

            for block in blocks:
                block_bbox = block[:4]
                text = block[4].strip()
                if not text:
                    continue

                in_table = False
                for t_bbox in table_bboxes:
                    if is_overlapping(block_bbox, t_bbox, threshold=0.4):
                        in_table = True
                        break

                if in_table:
                    continue

                elements.append(
                    (block_bbox[1], block_bbox[0], "text", text, block_bbox)
                )

            elements.sort(key=lambda x: (x[0], x[1]))

            page_contents = []
            for elem in elements:
                content = elem[3]
                page_contents.append(content)

            assembled_text = "\n\n".join(page_contents).strip()

            yield {"page_number": page_number, "text": assembled_text}

    except Exception as e:
        logger.error(f"Error parsing PDF file {file_path}: {e}", exc_info=True)
        raise e
    finally:
        if doc:
            doc.close()
            logger.info(f"Closed PDF file: {file_path}")
