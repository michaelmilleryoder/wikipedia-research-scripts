import csv
import re
import sys
import operator
import os
from collections import defaultdict
from IPython.core.debugger import Tracer; debug_here = Tracer()

csv.field_size_limit(sys.maxsize)

def wpsrc_regex():
    # Set source regex
    p = [r'\{\{((?:c|C)ite(?:\ |_)(?:book|journal|news|web).*?)\}\}']
    p.append(r'(<ref(?: name=.*?)?(?: ?\/>|>.*?<\/ref>))') # Probably would only need this one, but doesn't seem to double-count with line above
    p.append(r'(<ref(?: name=.*?)?(?:\/?>|>.*?<\/ref>))')
    wpsrc_regex = r'|'.join(p)
    return wpsrc_regex

def get_citations(intext):
    totalp = wpsrc_regex()
    citations = []
    for citation in re.findall(totalp, intext):
        for part in citation:
            if part != '':
                citations.append(part)
    return citations

def classify_citations(citations):
    isr_srcs = [['Jerusalem Post', 'jpost'],
                    ['Jewish Virtual Library', 'jewishvirtuallibrary'],
                    ['Aloni Shlomo', 'Shlomo'],
                    ['Avraham Sela', 'Sela'],
                    ['Anti-Defamation League', 'adl'],
                ['Menachem Klein', 'Klein', 'klein'],
                ['gov.il'],
                ['Jewish Journal', 'jewishjournal'],
                ['Ynetnews', 'ynetnews'],
                ['Intelligence and Terrorism Information Center', 'terrorism-info.org.il', 'intelligence.org.il']
                ]
    pal_srcs = [["Ma'an", 'Maan'],
                ["Finkelstein"],
                ["Human Rights Watch", 'hrw.org'],
               ["Amnesty International"],
               ["Edward Said", "Said"],
               ["B'Tselem", 'btselem'],
               ['Al-haq', 'alhaq'],
                ['Foundation for Middle East Peace', 'fmep'],
                ['Qumsiyeh', 'qumsiyeh'],
                ['Palestine Monitor', 'palestinemonitor'],
                ['Anti-War.com', 'anti-war.com'],
                ['Al-Ahram', 'al-ahram']
               ]

    isr_citations = []
    pal_citations = []
    neutral_citations = []

    for citation in citations:
        isr_match = False
        pal_match = False
        isrcite = None
        palcite = None

        found = False
        for isr_src in isr_srcs:
            for name in isr_src:
                if name in citation:
                    isr_match = True
                    isrcite = isr_src[0] # save src name
                    found = True
            if found:
                break
                    
        found = False
        for pal_src in pal_srcs:
            for name in pal_src:
                if name in citation:
                    pal_match = True
                    palcite = pal_src[0] # save src name
                    found = True
            if found:
                break
                
        if isr_match and pal_match:
            neutral_citations.append(citation)
        elif isr_match:
            isr_citations.append((isrcite,citation))
        elif pal_match:
#             debug_here()
            pal_citations.append((palcite,citation))
        else:
            neutral_citations.append(citation)

    return (isr_citations, pal_citations, neutral_citations)

# Get citations for raw revisions, keeping all unique citations ever added to that page and their count in all pp
rawdirpath = '/home/michael/school/cprose_research/wp/wp_articles/extract/extract_articles/ipc_articles_raw/'
srcpath = '/home/michael/school/cprose_research/wp/wp_articles/citations/article_citations.csv'

with open(srcpath, 'w') as out:
    writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['article_name','source', 'perspective', 'count'])
    
for p in sorted(os.listdir(rawdirpath)):
    artlines = []
    cite_info = defaultdict(int) # dict for citations for each article (citation, art_name): count
    rawpath = os.path.join(rawdirpath, p)
    with open(rawpath, 'r') as rawfile:
        reader = csv.reader(rawfile, delimiter='\t')
        for i, row in enumerate(reader): # for every revision
            art_name = row[0]
            ts = row[1]
            text = row[2]
            editor = row[3]
            citations = get_citations(text)
            if len(citations) > 0:
                isrc, palc, neutc = classify_citations(citations)
                if len(isrc) > 0 or len(palc) > 0:
                    for c in isrc:
                        cite_info[(art_name, c[0], 'israeli')] += 1
                    for c in palc:
                        cite_info[(art_name, c[0], 'palestinian')] += 1

    for c in cite_info:
        artlines.append(list(c) + [cite_info[c]])
        
    print('Finished with {:s}'.format(p))

    with open(srcpath, 'a') as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(artlines)
    
print("Wrote citations at {:s}".format(srcpath))
