import pandas as pd
import html5lib
import bs4
import sqlite3

# Install packages if needed on yen
# ! pip install --user html5lib
# ! pip install --user bs4

# Read S&P 500 companies CIK from wikipedia
table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",header =0, flavor = 'bs4')
table_1 = table[0]
duplicated_index = table[0][table[0]["CIK"].duplicated()].index
table_2 = table_1.drop(table_1.index[table[0][table[0]["CIK"].duplicated()].index])
ls_cik = table_2["CIK"]

# Connect to sqlite3 database
con = sqlite3.connect("/usr/local/ifs/gsb/EDGAR_HTTPS/edgar.db")
cur = con.cursor()

# Retrieve all the filenames, in a dictionary format with cik as key and filenames in list as value
dict_cik_filename = dict()
for i in ls_cik:
    ls_tmp = []
    cur.execute("select filename from edgar where cik = {0} and date >= '2010-01-01'and date <= '2019-12-31' and form = '10-K'  limit 30".format(i))
    result = cur.fetchall()
    for r in result:
        ls_tmp.append(r[0])
    dict_cik_filename[i] = ls_tmp


# Check file numbers
# ls_actual_num = [len(i) for i in dict_cik_filename.values()]
# sum(ls_actual_num)

# Combine filenames for different ciks
ls_files = []
for files in dict_cik_filename.values():
    ls_files = ls_files + files 

# See how's the filenames look like
# ls_files[:2]

path = "/usr/local/ifs/gsb/EDGAR_HTTPS"
dest = "/usr/local/ifs/gsb/usf_interns/Input10KFilesFolder/"
import shutil
src_files = ls_files
for file_name in src_files:
    full_file_name = path + "/" + file_name
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)


