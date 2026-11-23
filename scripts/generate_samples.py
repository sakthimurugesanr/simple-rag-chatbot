"""Generate tiny binary fixtures; requires requirements-dev.txt, no external services."""

from pathlib import Path


def generate_samples(destination: Path) -> None:
    """Create PDF/DOCX/XLSX/XLS fixtures with a shared, testable product fact."""
    import xlwt
    from docx import Document
    from openpyxl import Workbook
    from reportlab.pdfgen import canvas

    destination.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(destination / "policy.pdf"))
    for number in range(1, 4):
        pdf.drawString(72, 760, "Northstar internal handbook")
        pdf.drawString(72, 720, f"Policy page {number}")
        pdf.drawString(72, 690, "Annual leave allowance is 24 days.")
        pdf.drawString(72, 650, "Requests require manager approval.")
        pdf.drawString(72, 72, "Confidential - Northstar")
        pdf.showPage()
    pdf.save()
    blank = canvas.Canvas(str(destination / "scanned-placeholder.pdf"))
    blank.showPage()
    blank.save()
    doc = Document()
    doc.add_heading("Annual leave", level=1)
    doc.add_paragraph("Annual leave allowance is 24 days.")
    table = doc.add_table(rows=1, cols=3)
    for cell, value in zip(table.rows[0].cells, ["Product", "Price INR", "Stock"]):
        cell.text = value
    for cell, value in zip(table.add_row().cells, ["Atlas keyboard", "2500", "12"]):
        cell.text = value
    doc.add_heading("Remote work", level=1)
    doc.add_paragraph("Employees may work remotely 3 days per week.")
    doc.save(destination / "policy.docx")
    workbook = Workbook()
    products = workbook.active
    products.title = "Products"
    products.append(["Product", "Price INR", "Stock"])
    products.append(["Atlas keyboard", 2500, 12])
    products.append(["Orbit mouse", 1200, 25])
    staff = workbook.create_sheet("People")
    staff.append(["Name", "Team"])
    staff.append(["Mira Patel", "Engineering"])
    workbook.save(destination / "products.xlsx")
    workbook.close()
    legacy = xlwt.Workbook()
    sheet = legacy.add_sheet("Products")
    for row, values in enumerate(
        [
            ["Product", "Price INR", "Stock"],
            ["Atlas keyboard", 2500, 12],
        ]
    ):
        for column, value in enumerate(values):
            sheet.write(row, column, value)
    legacy.save(str(destination / "products.xls"))


if __name__ == "__main__":
    generate_samples(Path(__file__).resolve().parent.parent / "samples")
