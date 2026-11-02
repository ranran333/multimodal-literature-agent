"""Generate an entirely synthetic, redistributable two-page paper for demos/tests."""
import argparse
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

parser = argparse.ArgumentParser(description="Generate a fictional two-page demo PDF")
parser.add_argument("--force", action="store_true", help="Replace an existing demo PDF")
args = parser.parse_args()
path = Path("data/raw/synthetic-paper.pdf")
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists() and not args.force:
    print(f"{path} already exists; leaving it unchanged. Use --force to replace it.")
    raise SystemExit(0)
c = canvas.Canvas(str(path), pagesize=letter)
c.setTitle("Synthetic Study of Document Retrieval")
c.setFont("Helvetica-Bold", 17)
c.drawString(48, 740, "Synthetic Study of Document Retrieval")
c.setFont("Helvetica", 12)
c.drawString(48, 710, "Abstract: This fictional study compares retrieval systems on a toy corpus.")
c.drawString(48, 688, "The benchmark contains 120 research documents and 240 questions.")
c.drawString(48, 666, "The dataset is synthetic and is provided for a software demonstration.")
c.drawString(48, 644, "The evaluation metric is retrieval hit rate at five results.")
c.showPage()
c.setFont("Helvetica-Bold", 16)
c.drawString(48, 740, "Results and analysis")
c.setFont("Helvetica", 12)
c.drawString(48, 712, "Figure 1: Retrieval scores for Aspen and Cedar on the toy benchmark.")
c.drawString(48, 690, "The bars compare the two fictional systems; higher is better.")
c.setStrokeColor(colors.black)
c.line(100, 380, 100, 640)
c.line(100, 380, 490, 380)
c.setFillColor(colors.HexColor("#4C78A8"))
c.rect(165, 380, 90, 105, fill=1, stroke=0)
c.setFillColor(colors.HexColor("#F58518"))
c.rect(320, 380, 90, 200, fill=1, stroke=0)
c.setFillColor(colors.black)
c.drawString(180, 360, "Aspen")
c.drawString(335, 360, "Cedar")
c.save()
print(path)
