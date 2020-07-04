from bs4 import BeautifulSoup
import multiprocessing
import os
import re
import html

inputFolder = "DocumentsFolder/"
outputFolder = "ParsedDocumentsFolder/"

#To remove tables containing numeric content greater than 10 percent
def should_remove(text):
    if len(text) == 0:
        return True
    else:
        count_number = 0
        for c in text:
            if c.isnumeric():
                count_number = count_number + 1
        return count_number / len(text) > 0.10

#Parse 10K
def parse_10K(contents):

    documents = re.findall('(?s)<DOCUMENT>.*?</DOCUMENT>', contents)
    identifications = re.findall('<FILENAME>[^\s]+', contents)
    new_identifications = [i.replace("<FILENAME>", "") for i in identifications]
    for i in range(len(documents)):
        doc = documents[i]
        fileName = new_identifications[i]

        # Removal of pdf type documents named as 10-K
        if fileName.endswith(".pdf"):
            continue

        soup10k = BeautifulSoup(doc, 'html.parser')
        # Remove all anchor links from soup.
        [a.extract() for a in soup10k.find_all('a')]
        # Remove all numeric tables from soup.
        [s.extract() for s in soup10k.find_all('table') if should_remove(s.get_text().strip())]
        K10_text = soup10k.get_text(separator=" ")
        K10_text = K10_text.replace("\xa0", " ").replace("Table of Contents", " ").replace("_", "")

        # K10_text = K10_text.lower()
        # Remove whole text before PART 1-contain repeated info
        if "PART I" in K10_text:
            K10_text = K10_text.split("PART I", 1)[1]
        elif "PART\nI" in K10_text:
            K10_text = K10_text.split("PART\nI", 1)[1]

        # Remove whole text after PART IV-contain signatures and non-natural text
        if "PART IV" in K10_text:
            K10_text = K10_text.rsplit("PART IV", 1)[0]
        elif "PART\nIV" in K10_text:
            K10_text = K10_text.rsplit("PART\nIV", 1)[0]

    return K10_text

# Exhibits Parser
def parse_exhibits(contents):
    EX_text_whole = ""
    documents = re.findall('(?s)<DOCUMENT>.*?</DOCUMENT>', contents)
    for i, document in enumerate(documents):
        soup_exhibits = BeautifulSoup(document, 'lxml')
        [s.extract() for s in soup_exhibits.find_all('table') if should_remove(s.get_text().strip())]
        EX_text = soup_exhibits.text
        EX_text = EX_text.replace("\xa0", " ").replace("\'", "'").replace("Table of Contents", " ")
        # EX_text = EX_text.split("\n")[1]
        EX_text_whole += EX_text
    return EX_text

# XML Parser
def parse_xml(contents):
    xml_text = ""
    documents = re.findall('(?s)<DOCUMENT>.*?</DOCUMENT>', contents)
    for i, document in enumerate(documents):
        xml_without_head = re.sub(
            r'(?s)<head>.*?</head>', ' ', document, flags=re.IGNORECASE)
        soup_xml = BeautifulSoup(xml_without_head, "html.parser")
        [a.extract() for a in soup_xml.find_all('a')]
        [s.extract() for s in soup_xml.find_all('table')
         if should_remove(s.get_text().strip())]
        tmp = soup_xml.find_all(["font", "p"])
        s = ""
        for t in tmp:
            if 'Reference' not in str(t.text):
                s += t.text.replace("\xa0", "").replace("\xa030", "").replace(
                    "\\", "").replace("\n", " ").replace("No definition available.", " ") + " "
        xml_text += s
    return xml_text

fnames = os.listdir(inputFolder)
finished_files = os.listdir(outputFolder)
files_to_process = list(set(fnames) - set(finished_files))


def process_file(filename):
    print("Processing:", filename)
    logf = open("logFile.log", "a+")
    try:
        infile = open(inputFolder + filename, 'r')
        contents = infile.read()
        if filename.startswith("10-K"):
            parsedContent = parse_10K(contents)
        elif filename.startswith("XML"):
            parsedContent = parse_xml(contents)
        elif "EX-" in filename:
            parsedContent = parse_exhibits(contents)
        parsedContent=' '.join(parsedContent.split())
        writepath = outputFolder + "/parsed_" + filename
        mode = 'a' if os.path.exists(writepath) else 'w'
        with open(writepath, mode) as f:
            f.write(parsedContent)
    except Exception as e:
        logf.write("Error in processing file {0}: {1}\n".format(str(filename), str(e)))
    print("Processed:", filename)


with multiprocessing.Pool(5) as p:
    p.map(process_file, files_to_process)
