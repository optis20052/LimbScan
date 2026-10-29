"""Printable ArUco scale-marker sheets and marker definitions.

Every marker ID is unique, so the layout on the page doesn't matter for scale recovery:
only the printed side length does. Print at 100% ("actual size"), then check the 100 mm
ruler bar on the sheet with a real ruler.
"""
import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_100
MARKER_MM = 40.0          # printed black-square side length
SHEET_IDS = [list(range(0, 12)), list(range(12, 24)), list(range(24, 36))]


def dictionary():
    return cv2.aruco.getPredefinedDictionary(DICT)


def marker_image(mid, px):
    return cv2.aruco.generateImageMarker(dictionary(), mid, px)


def sheet_layout(sheet=0):
    """Return list of (id, x_mm, y_mm) marker top-left corners on an A4 portrait page (210x297)."""
    ids = SHEET_IDS[sheet]
    out = []
    cols, rows = 3, 4
    gx, gy = 65.0, 62.0
    x0 = (210 - (cols - 1) * gx - MARKER_MM) / 2
    y0 = 40.0
    for k, mid in enumerate(ids):
        r, c = divmod(k, cols)
        out.append((mid, x0 + c * gx, y0 + r * gy))
    return out


def render_sheet(sheet=0, dpmm=12):
    """Rasterise an A4 sheet (white page) as a uint8 image at dpmm pixels per mm."""
    W, H = int(210 * dpmm), int(297 * dpmm)
    img = np.full((H, W), 255, np.uint8)
    s = int(round(MARKER_MM * dpmm))
    for mid, x, y in sheet_layout(sheet):
        # 1-cell quiet zone is the white page itself
        m = marker_image(mid, s)
        X, Y = int(round(x * dpmm)), int(round(y * dpmm))
        img[Y:Y + s, X:X + s] = m
    return img


def write_pdf(path):
    """Vector PDF with exact physical dimensions (no rasterisation scale ambiguity)."""
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=(210 * mm, 297 * mm))
    d = dictionary()
    for sheet in range(len(SHEET_IDS)):
        c.setFont('Helvetica-Bold', 11)
        c.drawString(15 * mm, 285 * mm, f'LimbScan scale sheet {sheet + 1}/{len(SHEET_IDS)}  -  print at 100% / "Actual size"')
        c.setFont('Helvetica', 9)
        c.drawString(15 * mm, 279 * mm, f'Each black square must measure exactly {MARKER_MM:.0f}.0 mm. Check the bar below with a ruler.')
        # 100 mm check bar
        c.setLineWidth(0.6)
        c.line(55 * mm, 15 * mm, 155 * mm, 15 * mm)
        for i in range(11):
            h = 4 if i % 5 == 0 else 2
            c.line((55 + 10 * i) * mm, 15 * mm, (55 + 10 * i) * mm, (15 + h) * mm)
        c.drawString(90 * mm, 21 * mm, '100 mm check bar')
        bits = d.bytesList  # draw cells as vector rectangles
        n = d.markerSize + 2
        cell = MARKER_MM / n
        for mid, x, y in sheet_layout(sheet):
            m = marker_image(mid, n * 10)[5::10, 5::10]
            for r in range(n):
                for col in range(n):
                    if m[r, col] < 128:
                        X = (x + col * cell) * mm
                        Y = (297 - y - (r + 1) * cell) * mm
                        c.rect(X, Y, cell * mm + 0.01, cell * mm + 0.01, stroke=0, fill=1)
            c.setFont('Helvetica', 6)
            c.drawString(x * mm, (297 - y - MARKER_MM - 4) * mm, f'#{mid}')
        c.showPage()
    c.save()
