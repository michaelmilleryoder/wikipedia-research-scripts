import csv
import re
import sys
import os
import numpy as np

from collections import defaultdict, Counter
import pdb

csv.field_size_limit(sys.maxsize)

p = [r'\{\{((?:c|C)ite(?:\ |_)(?:book|journal|news|web).*?)\}\}']
p.append(r'<ref(?: name=([^<]*?))?>([^<]*?)<\/ref>|<ref name=([^>]*?)\/?>') # Probably would only need this one, but doesn't seem to double-count with line above
wpsrc_regex = r'|'.join(p)

def get_citations(intext):
    totalp = wpsrc_regex
    citations = []
    for citation in re.findall(totalp, intext):
        for part in citation:
            if part != '':
                citations.append(part)
    return citations

rawdirpath = '/home/michael/school/research/wp/wp_articles/extract/extract_articles/ipc_articles_raw/'
outdirpath = '/home/michael/school/research/wp/wp_articles/citations/'

for p in sorted(os.listdir(rawdirpath)):
    outpath = os.path.join(outdirpath, p[:-4] + '_citations.csv')
    rawpath = os.path.join(rawdirpath, p)
    with open(rawpath, 'r') as rawfile:
        art_info = []
        reader = csv.reader(rawfile, delimiter='\t')
        for i, row in enumerate(reader):
            art_name = row[0]
            ts = row[1]
            text = row[2]
            citations = get_citations(text)
            art_info += [[art_name, ts, c] for c in citations]

    # Write csv
    with open(outpath, 'w') as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['article_name', 'timestamp', 'citation'])
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(art_info)

    print("Wrote csv at", outpath)
