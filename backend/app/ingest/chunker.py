import re
from typing import TypedDict


class DocumentChunkData(TypedDict):
    chunk_index: int
    page_number: int
    section_name: str | None
    text_content: str


MARKDOWN_HEADER_REGEX = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_DELIMITER_REGEX = re.compile(r"^\s*\|?\s*(?:-{1,}:?\s*\|)+\s*(?:-{1,}:?\s*)?\|?\s*$")
CODE_FENCE_REGEX = re.compile(r"^\s*(?:`{3,}|~{3,})")


def parse_markdown_header(line: str) -> tuple[int, str] | None:
    match = MARKDOWN_HEADER_REGEX.match(line.strip())
    if match:
        level = len(match.group(1))
        title = match.group(2).strip()
        if 1 <= len(title) <= 150:
            return level, title
    return None


def update_header_stack(
    stack: list[tuple[int, str]], level: int, title: str
) -> list[tuple[int, str]]:
    return [item for item in stack if item[0] < level] + [(level, title)]


def format_breadcrumbs(stack: list[tuple[int, str]]) -> str | None:
    return " > ".join(title for _, title in stack) if stack else None


def is_table_delimiter_row(line: str) -> bool:
    return bool(TABLE_DELIMITER_REGEX.match(line.strip()))


def is_table_row_candidate(line: str) -> bool:
    """
    Tightened table row check:
    Ensures the line is structured like a Markdown table row and does not
    mistake regular prose containing a single '|' (such as '|' used as 'or') as a table.
    Requires:
      1. Starting and ending with '|' (with at least 1 character inside), OR
      2. At least two '|' characters with spacing or columns.
    """
    s = line.strip()
    if len(s) < 2 or "|" not in s:
        return False
    if s.startswith("|") and s.endswith("|"):
        return True
    return s.count("|") >= 2


def split_large_table(table_text: str, max_size: int = 1200) -> list[str]:
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if len(lines) <= 2 or len(table_text) <= max_size:
        return [table_text.strip()]

    header_block = ""
    data_rows = lines
    # Explicitly validate:
    # 1. line 1 is a valid delimiter row
    # 2. line 0 is a valid table row candidate
    # 3. line 0 is NOT itself a delimiter row
    if (
        len(lines) > 1
        and is_table_delimiter_row(lines[1])
        and is_table_row_candidate(lines[0])
        and not is_table_delimiter_row(lines[0])
    ):
        header_block = f"{lines[0]}\n{lines[1]}\n"
        data_rows = lines[2:]

    chunks: list[str] = []
    current_chunk = header_block

    for row in data_rows:
        row_str = row + "\n"
        if len(current_chunk) + len(row_str) > max_size and current_chunk != header_block:
            chunks.append(current_chunk.strip())
            current_chunk = header_block + row_str
        else:
            current_chunk += row_str

    if current_chunk.strip() != header_block.strip():
        chunks.append(current_chunk.strip())

    return chunks


def parse_page_into_blocks(page_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_type: str | None = None
    current_lines: list[str] = []
    in_code_fence = False

    def flush():
        nonlocal current_type, current_lines
        if current_lines:
            blocks.append((current_type or "text", "\n".join(current_lines).strip()))
            current_lines = []
            current_type = None

    lines = page_text.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()

        # Phase 1.1: Fenced Code Block Tracking (``` or ~~~)
        if CODE_FENCE_REGEX.match(stripped):
            if not in_code_fence:
                # Entering code fence: flush any preceding block and switch to dedicated code block
                flush()
                in_code_fence = True
                current_type = "code"
                current_lines.append(raw_line)
                i += 1
                continue
            else:
                # Exiting code fence: append closing fence and flush atomically
                current_lines.append(raw_line)
                flush()
                in_code_fence = False
                i += 1
                continue

        # Inside a code fence: bypass all markdown parsing, regexes, and blank line breaks
        if in_code_fence:
            current_lines.append(raw_line)
            i += 1
            continue

        # Empty line outside code fence always terminates the current block
        if not stripped:
            if current_type != "table":
                flush()
            i += 1
            continue

        if parse_markdown_header(stripped):
            flush()
            blocks.append(("heading", stripped))
            i += 1
            continue

        # Phase 1.2 & 1.3: Table lookahead with tightened candidate detection
        if (
            current_type != "table"
            and is_table_row_candidate(stripped)
            and not is_table_delimiter_row(stripped)
            and i + 1 < n
            and is_table_delimiter_row(lines[i + 1])
        ):
            flush()
            current_type = "table"
            current_lines.append(stripped)
            current_lines.append(lines[i + 1].strip())
            i += 2
            continue

        if current_type == "table":
            if is_table_row_candidate(stripped):
                current_lines.append(stripped)
                i += 1
                continue
            else:
                flush()

        current_type = "text"
        current_lines.append(raw_line)
        i += 1

    flush()
    return blocks


def extract_word_overlap(text: str, target_overlap_chars: int) -> str:
    """
    Phase 2: Index-based backward traversal.
    Directly slices the original string without calling .split() or ' '.join(),
    preserving structural whitespace, indentation, tabs, and line breaks intact.
    """
    if target_overlap_chars <= 0 or not text:
        return ""

    text_len = len(text)
    if text_len <= target_overlap_chars:
        return text

    # Slicing window start candidate
    slice_start = text_len - target_overlap_chars

    # Scan forward to find the nearest whitespace boundary to avoid word cutting
    boundary_idx = slice_start
    while boundary_idx < text_len and not text[boundary_idx].isspace():
        boundary_idx += 1

    # Skip past the whitespace itself to start cleanly at the next word
    while boundary_idx < text_len and text[boundary_idx].isspace():
        boundary_idx += 1

    # If scan reached the end without finding a split point, fallback to backward search
    if boundary_idx >= text_len:
        boundary_idx = slice_start
        while boundary_idx > 0 and not text[boundary_idx].isspace():
            boundary_idx -= 1
        while boundary_idx < text_len and text[boundary_idx].isspace():
            boundary_idx += 1

    return text[boundary_idx:]


def split_oversized_text(text: str, max_size: int) -> list[str]:
    """
    Phase 3: Custom boundary-aware safe text splitter.
    Does NOT use textwrap.wrap, preserving indentation, lists, code blocks,
    and internal whitespace.
    Splits along the natural hierarchy: '\\n', then '. ', then ' '.
    """
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_size:
        window = remaining[:max_size]
        split_pos = -1

        # 1. Try splitting at newline boundary
        newline_pos = window.rfind("\n")
        if newline_pos > max_size // 3:
            split_pos = newline_pos + 1
        else:
            # 2. Try splitting at sentence boundary ('. ')
            dot_pos = window.rfind(". ")
            if dot_pos > max_size // 3:
                split_pos = dot_pos + 2
            else:
                # 3. Try splitting at whitespace boundary (' ')
                space_pos = window.rfind(" ")
                if space_pos > max_size // 3:
                    split_pos = space_pos + 1
                else:
                    # 4. Hard fallback if no whitespace found
                    split_pos = max_size

        chunk = remaining[:split_pos]
        chunks.append(chunk)
        remaining = remaining[split_pos:]

    if remaining:
        chunks.append(remaining)

    return chunks


def chunk_page(
    page_text: str,
    page_number: int,
    start_chunk_idx: int,
    header_stack: list[tuple[int, str]] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[DocumentChunkData], list[tuple[int, str]]]:
    """
    Strict, structure-aware Markdown chunker without illusions or magic numbers.
    Preserves fenced code blocks, table integrity, and exact whitespace layout.
    """
    active_stack: list[tuple[int, str]] = list(header_stack) if header_stack else []
    if not page_text or not page_text.strip():
        return [], active_stack

    blocks = parse_page_into_blocks(page_text)
    chunks: list[DocumentChunkData] = []
    chunk_idx = start_chunk_idx

    current_chunk_text = ""
    last_content_type: str | None = None

    safe_wrap_width = max(100, chunk_size - chunk_overlap)

    def emit_chunk() -> None:
        nonlocal chunk_idx, current_chunk_text, last_content_type
        if current_chunk_text.strip():
            chunks.append(
                {
                    "chunk_index": chunk_idx,
                    "page_number": page_number,
                    "section_name": format_breadcrumbs(active_stack),
                    "text_content": current_chunk_text.strip(),
                }
            )
            chunk_idx += 1
            current_chunk_text = ""
            last_content_type = None

    for block_type, content in blocks:
        if block_type == "heading":
            parsed = parse_markdown_header(content)
            if parsed:
                if last_content_type and last_content_type != "heading":
                    emit_chunk()
                active_stack = update_header_stack(active_stack, parsed[0], parsed[1])

            current_chunk_text += content + "\n\n"
            last_content_type = "heading"

        elif block_type == "table":
            table_slices = split_large_table(content, max_size=chunk_size)
            for t_slice in table_slices:
                if len(current_chunk_text) + len(t_slice) > chunk_size and current_chunk_text.strip():
                    emit_chunk()

                sep = "\n\n" if current_chunk_text.strip() else ""
                current_chunk_text += f"{sep}{t_slice}\n\n"
                last_content_type = "table"

        elif block_type == "code":
            # Code blocks must NOT be split by paragraph (\n\n) or have indentation stripped
            code_units = [content] if len(content) <= chunk_size else split_oversized_text(content, chunk_size)
            for c_unit in code_units:
                if len(current_chunk_text) + len(c_unit) > chunk_size and current_chunk_text.strip():
                    emit_chunk()

                sep = "\n\n" if current_chunk_text.strip() else ""
                current_chunk_text += f"{sep}{c_unit}\n\n"
                last_content_type = "code"

        else:
            paragraphs = [p for p in content.split("\n\n") if p.strip()]

            for para in paragraphs:
                if len(para) <= safe_wrap_width:
                    units = [para]
                else:
                    units = split_oversized_text(para, safe_wrap_width)

                for unit_idx, unit in enumerate(units):
                    if len(current_chunk_text) + len(unit) > chunk_size and current_chunk_text.strip():
                        overlap_seed = ""
                        if last_content_type == "text" and chunk_overlap > 0:
                            overlap_seed = extract_word_overlap(current_chunk_text, chunk_overlap)

                        emit_chunk()

                        if overlap_seed:
                            current_chunk_text = f"{overlap_seed} {unit}"
                        else:
                            current_chunk_text = unit
                        last_content_type = "text"
                    else:
                        if not current_chunk_text.strip():
                            current_chunk_text = unit
                        elif last_content_type == "text" and len(units) > 1 and unit_idx > 0:
                            current_chunk_text += f" {unit}"
                        else:
                            sep = (
                                ""
                                if current_chunk_text.endswith("\n\n")
                                else ("\n" if current_chunk_text.endswith("\n") else "\n\n")
                            )
                            current_chunk_text += f"{sep}{unit}"
                        last_content_type = "text"

    emit_chunk()
    return chunks, active_stack