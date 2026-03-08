from pdf_loader import load_pdf,chunk_text
pdf_path="paper.pdf"

text=load_pdf(pdf_path)

chunks=chunk_text(text)

print("PDF loaded")
print("Total chunks:",len(chunks))