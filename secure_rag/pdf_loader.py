from pypdf import PdfReader #library to read pdf files


def load_pdf(file_path):#defined fn which takes file_path as input
    reader = PdfReader(file_path)#loading pdf
    text = ""

    for page in reader.pages:#looping through pages
        extracted = page.extract_text()#extract text
        if extracted:#validating text
            text += extracted + "\n"#appending to new line (preserving page seperation)

    return text


def chunk_text(text, chunk_size=500):#breaks text into chunks
    words = text.split()#split into word
    chunks = []#empty array to store chunks
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks
