import os
import re
import csv
from PyPDF2 import PdfReader

pdf_folder = "C:\Users\admin\Desktop\wolachat\wolachat"  
csv_output = "wolaytta_dictionary.csv"
txt_output = "wolaytta_chunks.txt"

all_text = ""
for file in sorted(os.listdir(pdf_folder)):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
                
entries = []
chunks = []

pattern = r"(?P<word>\w+)%\d+\. (?P<pos>\w+)%(?P<content>.*?)((?=\n\w+%)|$)"

for match in re.finditer(pattern, all_text, re.DOTALL):
    word = match.group("word").strip()
    pos = match.group("pos").strip()
    content = match.group("content").strip()
    senses = content.split("►")
    for sense in senses:
        if not sense.strip():
            continue
        parts = sense.strip().split("●")
        meaning = parts[0].strip()
        wolaytta_example = parts[1].strip() if len(parts) > 1 else ""
        eng_example = ""
        if "%○" in wolaytta_example:
            wolaytta_example, eng_example = wolaytta_example.split("%○", 1)
            wolaytta_example = wolaytta_example.strip()
            eng_example = eng_example.strip()

        entries.append([word, pos, meaning, wolaytta_example, eng_example])
        chunk = f"{word} ({pos}): {meaning}\nExample (Wolaytta): {wolaytta_example}"
        if eng_example:
            chunk += f"\nExample (English): {eng_example}"
        chunks.append(chunk.strip())

# === 4. Save CSV ===
with open(csv_output, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Word", "Part of Speech", "Meaning", "Example (Wolaytta)", "Example (English)"])
    writer.writerows(entries)

# === 5. Save TXT chunks ===
with open(txt_output, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n\n")

print(f"✅ Done! Saved {len(entries)} entries to CSV and {len(chunks)} chunks for RAG.")