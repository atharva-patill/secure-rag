from .pypdf import PdfReader #library to read pdf files


def load_pdf(file_path):#defined fn which takes file_path as input
    reader = PdfReader(file_path)#loading pdf
    text = ""

    for page in reader.pages:#looping through pages
        extracted = page.extract_text()#extract text
        if extracted:#validating text
            text += extracted + "\n"#appending to new line (preserving page seperation)

    return text


def chunk_text(text, chunk_size=500,overlap=50):#breaks text into chunks
    words = text.split()#split into word
    chunks = []#empty array to store chunks
    step=chunk_size-overlap
    if step<=0:
        raise ValueError("overlap must be smaller than chunk_size")
    for i in range(0, len(words), step):#jumps every 500 words (because each chunk == 500 words)
        chunk = " ".join(words[i : i + chunk_size])#converting back to text
        if chunk:
            chunks.append(chunk)
    return chunks
