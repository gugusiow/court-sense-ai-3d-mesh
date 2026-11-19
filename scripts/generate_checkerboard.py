"""Write a 4-page A4 checkerboard that tapes into one calibration board.

The pattern is 6 by 5 squares of 50 mm. That is 5 by 4 inner corners, which
is what calibrate_stereo.py expects as --board-cols 5 --board-rows 4.
Squares are drawn as vectors, so a printer set to 100% keeps the size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MM = 72 / 25.4
PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
TRIM_MM = 10.0
TILE_W_MM = PAGE_W_MM - 2 * TRIM_MM
TILE_H_MM = PAGE_H_MM - 2 * TRIM_MM
SQUARES_X = 6
SQUARES_Y = 5
SQUARE_MM = 50.0
INNER_COLS = SQUARES_X - 1
INNER_ROWS = SQUARES_Y - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable A4 checkerboard.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/checkerboard_a4.pdf"),
    )
    args = parser.parse_args()
    board = _board_geometry()
    pdf = _render_pdf(board)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pdf)
    _check_detection(board)
    print(f"wrote {args.output}")
    print(
        f"squares {SQUARE_MM:.0f} mm, inner corners {INNER_COLS} x {INNER_ROWS}. "
        "Print at 100% scale, cut the dashed lines, and tape A B over C D."
    )


def _board_geometry() -> dict:
    board_w = TILE_W_MM * 2
    board_h = TILE_H_MM * 2
    pattern_w = SQUARES_X * SQUARE_MM
    pattern_h = SQUARES_Y * SQUARE_MM
    # Keep the page seams on square edges so a taped join does not cross a square.
    pattern_x = TILE_W_MM - 3 * SQUARE_MM
    pattern_y = TILE_H_MM - 3 * SQUARE_MM
    return {
        "board_w": board_w,
        "board_h": board_h,
        "pattern_x": pattern_x,
        "pattern_y": pattern_y,
        "pattern_w": pattern_w,
        "pattern_h": pattern_h,
    }


def _render_pdf(board: dict) -> bytes:
    pages = [_page_stream(col, row, board) for row in range(2) for col in range(2)]
    objects: list[bytes] = []

    def add(data: str) -> int:
        objects.append(data.encode("latin-1"))
        return len(objects)

    font_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    content_ids = []
    for stream in pages:
        content_ids.append(
            add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        )
    # Pages object is created after page objects so the kids list can point at them.
    pages_id_placeholder = len(objects) + 1 + len(content_ids)
    for content_id in content_ids:
        page_ids.append(
            add(
                "<< /Type /Page /Parent {pages} 0 R "
                "/MediaBox [0 0 {w:.2f} {h:.2f}] "
                "/Resources << /Font << /F1 {font} 0 R >> >> "
                "/Contents {content} 0 R >>".format(
                    pages=pages_id_placeholder,
                    w=PAGE_W_MM * MM,
                    h=PAGE_H_MM * MM,
                    font=font_id,
                    content=content_id,
                )
            )
        )
    pages_id = add(
        "<< /Type /Pages /Count {count} /Kids [{kids}] >>".format(
            count=len(page_ids),
            kids=" ".join(f"{page_id} 0 R" for page_id in page_ids),
        )
    )
    if pages_id != pages_id_placeholder:
        raise RuntimeError("PDF page tree id drifted")
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")
    return _serialize(objects, catalog_id)


def _page_stream(col: int, row: int, board: dict) -> str:
    tile = (col * TILE_W_MM, row * TILE_H_MM, (col + 1) * TILE_W_MM, (row + 1) * TILE_H_MM)
    commands = ["0 0 0 rg", "0.2 w"]
    for square in _black_squares(board):
        clipped = _intersect(square, tile)
        if clipped is not None:
            commands.append(_fill_rect(*_to_page(clipped, col, row)))
    commands.append("0.4 w [4 3] 0 d")
    commands.append(_stroke_rect(*_to_page((tile[0], tile[1], tile[2], tile[3]), col, row)))
    commands.append("[] 0 d")
    commands.extend(_crop_marks(col, row))
    commands.extend(_labels(col, row, board))
    return "\n".join(commands) + "\n"


def _black_squares(board: dict) -> list[tuple[float, float, float, float]]:
    squares = []
    for row in range(SQUARES_Y):
        for col in range(SQUARES_X):
            if (col + row) % 2 != 0:
                continue
            x0 = board["pattern_x"] + col * SQUARE_MM
            y0 = board["pattern_y"] + row * SQUARE_MM
            squares.append((x0, y0, x0 + SQUARE_MM, y0 + SQUARE_MM))
    return squares


def _labels(col: int, row: int, board: dict) -> list[str]:
    letter = "ABCD"[row * 2 + col]
    lines = [f"{letter}"]
    if (col, row) == (0, 0):
        lines.extend(
            [
                "CourtSense checkerboard",
                "Square size: 50 mm",
                "Inner corners: 5 across, 4 down",
                "Print at 100 percent, not fit-to-page",
                "Cut the dashed line. Tape A B",
                "on the top row and C D below.",
                "Keep the sheet flat.",
            ]
        )
    if (col, row) == (1, 1):
        lines.extend(
            [
                "This bar must measure 50 mm.",
                "square-mm 50",
                "board-cols 5",
                "board-rows 4",
            ]
        )
    text = []
    cursor_y = 18 if row == 0 else TILE_H_MM - 70
    for index, line in enumerate(lines):
        size = 18 if index == 0 else 9
        text.append(_text(12, cursor_y, line, size, col, row))
        cursor_y += 6 if index else 10
    if (col, row) == (1, 1):
        bar_y = 2 * TILE_H_MM - 24
        bar = (
            TILE_W_MM + 24,
            bar_y,
            TILE_W_MM + 24 + SQUARE_MM,
            bar_y + 6,
        )
        clipped = _intersect(bar, (TILE_W_MM, TILE_H_MM, 2 * TILE_W_MM, 2 * TILE_H_MM))
        if clipped is not None:
            text.append(_fill_rect(*_to_page(clipped, col, row)))
    return text


def _crop_marks(col: int, row: int) -> list[str]:
    marks = []
    inset = TRIM_MM * MM
    gap = 1.2 * MM
    length = 6 * MM
    page_w = PAGE_W_MM * MM
    page_h = PAGE_H_MM * MM
    corners = (
        (inset, inset),
        (page_w - inset, inset),
        (inset, page_h - inset),
        (page_w - inset, page_h - inset),
    )
    for x, y in corners:
        horizontal = -1 if x < page_w / 2 else 1
        vertical = -1 if y < page_h / 2 else 1
        marks.append(
            "{x0:.2f} {y:.2f} m {x1:.2f} {y:.2f} l S".format(
                x0=x + horizontal * gap,
                x1=x + horizontal * (gap + length),
                y=y,
            )
        )
        marks.append(
            "{x:.2f} {y0:.2f} m {x:.2f} {y1:.2f} l S".format(
                x=x,
                y0=y + vertical * gap,
                y1=y + vertical * (gap + length),
            )
        )
    return marks


def _text(x_mm: float, y_mm: float, value: str, size: float, col: int, row: int) -> str:
    # y_mm is measured down from the top of this tile's cut area.
    page_x = (TRIM_MM + x_mm) * MM
    page_y = (PAGE_H_MM - TRIM_MM - y_mm) * MM
    safe = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /F1 {size:.1f} Tf {page_x:.2f} {page_y:.2f} Td ({safe}) Tj ET"


def _to_page(rect: tuple[float, float, float, float], col: int, row: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    page_x0 = TRIM_MM + (x0 - col * TILE_W_MM)
    page_x1 = TRIM_MM + (x1 - col * TILE_W_MM)
    top = TRIM_MM + (y0 - row * TILE_H_MM)
    bottom = TRIM_MM + (y1 - row * TILE_H_MM)
    pdf_y0 = PAGE_H_MM - bottom
    pdf_y1 = PAGE_H_MM - top
    return page_x0 * MM, pdf_y0 * MM, (page_x1 - page_x0) * MM, (pdf_y1 - pdf_y0) * MM


def _fill_rect(x: float, y: float, w: float, h: float) -> str:
    return f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f"


def _stroke_rect(x: float, y: float, w: float, h: float) -> str:
    return f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re S"


def _intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _serialize(objects: list[bytes], catalog_id: int) -> bytes:
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)


def _check_detection(board: dict) -> None:
    import cv2

    pixels_per_mm = 4
    height = int(round(board["board_h"] * pixels_per_mm))
    width = int(round(board["board_w"] * pixels_per_mm))
    image = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in _black_squares(board):
        image[
            int(round(y0 * pixels_per_mm)) : int(round(y1 * pixels_per_mm)),
            int(round(x0 * pixels_per_mm)) : int(round(x1 * pixels_per_mm)),
        ] = 0
    found, corners = cv2.findChessboardCorners(image, (INNER_COLS, INNER_ROWS))
    if not found or corners is None or len(corners) != INNER_COLS * INNER_ROWS:
        raise RuntimeError("OpenCV did not find the generated checkerboard")


if __name__ == "__main__":
    main()
