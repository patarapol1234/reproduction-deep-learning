#!/usr/bin/env python3
"""Render the report with PyMuPDF, without launching a browser."""
import argparse
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "report")
    args = parser.parse_args()
    import fitz
    directory = args.report_dir.resolve()
    html = (directory / "official-tent-report.html").read_text()
    # Override box backgrounds to avoid cross-page painting artifacts in MuPDF.
    css = "body { margin: 0; max-width: none; } h1 { margin-top: 0; } .status, th { background-color: transparent; }"
    story = fitz.Story(html=html, user_css=css, archive=fitz.Archive(str(directory)), em=10)
    buffer = io.BytesIO()
    writer = fitz.DocumentWriter(buffer)
    page = fitz.paper_rect("a4")
    rectangle = fitz.Rect(51, 42, page.width - 51, page.height - 42)
    more = True
    count = 0
    while more:
        device = writer.begin_page(page)
        more, filled = story.place(rectangle)
        story.draw(device)
        writer.end_page()
        count += 1
        if count > 20:
            raise RuntimeError("Unexpected report overflow")
    writer.close()
    document = fitz.open(stream=buffer.getvalue(), filetype="pdf")
    for index, p in enumerate(document):
        p.insert_text((51, page.height - 27), f"Official Tent v2 | {index + 1} / {len(document)}",
                      fontsize=8, fontname="helv", color=(.35, .42, .47))
    document.set_metadata({"title": "Reproducing the Official Tent Example — v2",
                           "subject": "Official-code experiment preparation, execution evidence or measured results"})
    output = directory / "official-tent-report.pdf"
    document.save(output, garbage=4, deflate=True)
    text = "\n".join(p.get_text() for p in document)
    status = json.loads((directory / "report-status.json").read_text())
    if status["status"] == "prepared" and "pending" not in text.lower():
        raise RuntimeError("Preparation status is missing from the rendered PDF")
    if status["status"] == "interrupted" and ("Interrupted before evaluation" not in text or "Not measured" not in text):
        raise RuntimeError("Interruption status or unavailable measurements are missing from the PDF")
    if "6. Conclusion" not in text or "References" not in text:
        raise RuntimeError("Report content is incomplete in the PDF")
    (directory / "pdf-quality.json").write_text(json.dumps({
        "pages": len(document), "bytes": output.stat().st_size,
        "status": status["status"], "text_complete": True,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
