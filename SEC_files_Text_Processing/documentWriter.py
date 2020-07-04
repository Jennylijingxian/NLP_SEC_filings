from bs4 import BeautifulSoup
import multiprocessing
import os
import re

inputFolder = "Input10KFilesFolder/"
outputFolder = "DocumentsFolder/"


fnames = os.listdir(inputFolder)
finished_files = os.listdir(outputFolder)
files_to_process = list(set(fnames) - set(finished_files))


def process_file(filename):
    print("Processing:", filename)
    infile = open(inputFolder + filename, 'r')
    contents = infile.read()
    documents = re.findall('(?s)<DOCUMENT>.*?</DOCUMENT>', contents)
    identifications = re.findall('<TYPE>[^\s]+', contents)
    new_identifications = [i.replace("<TYPE>", "") for i in identifications]


    for i in range(len(documents)):
        typeDoc = new_identifications[i]
        doc = documents[i]
        excluded_file_formats = ["EXCEL", "JSON","ZIP", "GRAPHIC", "PDF", "EX-101.INS", "EX-101.SCH", "EX-101.CAL",
                                 "EX-101.DEF", "EX-101.LAB","EX-101.PRE"]
        if typeDoc in excluded_file_formats:
            continue
        writepath = outputFolder+"/"+typeDoc+"_"+filename
        mode = 'a' if os.path.exists(writepath) else 'w'
        with open(writepath, mode) as f:
            f.write(str(doc))
    print("Processed:", filename)

with multiprocessing.Pool(9) as p:
    p.map(process_file, files_to_process)
