from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Paths to the images you created earlier
image_paths = [
    "backend/app/services/scanned_page1.png",
    "backend/app/services/scanned_page2.png"
]

output_pdf = "backend/app/services/dummy_scanned.pdf"
c = canvas.Canvas(output_pdf, pagesize=letter)

for img_path in image_paths:
    # Draw the image on the PDF page
    c.drawImage(img_path, 0, 500, width=400, height=100)
    c.showPage()  # Add a new page for the next image

c.save()
print(f"Created {output_pdf} with {len(image_paths)} image pages.")