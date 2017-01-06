import csv
import re
import sys
import operator
import os
import numpy as np
import hashlib

from collections import defaultdict, Counter

csv.field_size_limit(sys.maxsize)

p = [r'\{\{((?:c|C)ite(?:\ |_)(?:book|journal|news|web).*?)\}\}']
p.append(r'<ref(?: name=([^<]*?))?>([^<]*?)<\/ref>|<ref name=([^>]*?)\/?>') # Probably would only need this one, but doesn't seem to double-count with line above
wpsrc_regex = r'|'.join(p)

def get_citations(intext):
    # p0 = r'TEMPLATE\[((?:c|C)ite(?:\ |_)(?:book|journal|news|web).*?)\]'
    #p1 = r'|\{\{(cite (?:book|journal|news|web).*?=\ )'
    #p.append(r'(<ref(?: name=.*?)?>.*?<\/ref>)') # Probably would only need this one, but doesn't seem to double-count with line above
    totalp = wpsrc_regex
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
    
        for isr_src in isr_srcs:
            if any(name in citation for name in isr_src):
                isr_match = True
                break
        for pal_src in pal_srcs:
            if any(name in citation for name in pal_src):
                pal_match = True
                break
        if isr_match and pal_match:
            neutral_citations.append(citation)
        elif isr_match:
            isr_citations.append(citation)
        elif pal_match:
            pal_citations.append(citation)
        else:
            neutral_citations.append(citation)
    
    return (isr_citations, pal_citations, neutral_citations)


def edit_length(previous_text, current_text):
    cprev = Counter(previous_text.split(' '))
    ccur = Counter(current_text.split(' '))
    diff0 = cprev-ccur
    diff1 = ccur-cprev
    return sum(diff0.values()) + sum(diff1.values())


# Get citations for raw revisions, construct csv with citation bias values
vandalism = "vandal"
revert = "revert"
token = "(?:^|.*\s)(?:{0})(?:$|\s)"
rvv = token.format("rvv")
rv = token.format("rv")
vandal_matches = re.compile("{0}|{1}".format(vandalism,rvv),re.IGNORECASE)
revert_matches = re.compile("{0}|{1}".format(revert,rv),re.IGNORECASE)

rawdirpath = '/home/michael/school/cprose_research/wp/wp_articles/extract/extract_articles/ipc_articles_raw/'
#rawpath = '/home/michael/school/cprose_research/wp/wp_articles/extract/extract_articles/ipc_articles_raw/second_intifada.tsv'
outpath = '/home/michael/school/cprose_research/wp/wp_articles/citations_reverts.csv'

with open(outpath, 'w') as out:
    writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['article_name', 'timestamp', 'editor', 'edit_comment',
                     'israeli_citations', 'palestinian_citations', 'neutral_citations',
                    'edit_length', 'revert_md5', 'revert_comment', 'vandalism_comment'])
    
for p in os.listdir(rawdirpath):
    hashes = set()
    art_info = []
    rawpath = os.path.join(rawdirpath, p)
    with open(rawpath, 'r') as rawfile:
        reader = csv.reader(rawfile, delimiter='\t')
        prev_text = None
        for i, row in enumerate(reader):
            art_name = row[0]
            ts = row[1]
            text = row[2]
            editor = row[3]
            comment = row[4]
            citations = get_citations(text)
            n_isr = 0
            n_pal = 0
            n_neut = 0
            if len(citations) > 0:
                isrc, palc, neutc = classify_citations(citations)
                n_isr, n_pal, n_neut = len(isrc), len(palc), len(neutc)
                
            # Get edit length
            if prev_text:
                el = edit_length(prev_text, text)
            else:
                el = len(text.split(' '))
            
            # Get revert info
            m = hashlib.md5()
            #m.update(text.encode('utf8'))
            m.update(text)
            
            #tag mentions vandalism
            rvv = (1 if vandal_matches.match(comment) else 0)
                   
            #tag mentions revert
            rv = (1 if revert_matches.match(comment) else 0)
            
            #restores previous version
            md5 = (1 if m.digest() in hashes else 0)
            
            if not md5:
                hashes.add(m.digest())
            
            prev_text = text
            art_info.append([art_name, ts, editor, comment, n_isr, n_pal, n_neut, el, md5, rv, rvv])
    print(p)

    # Write csv
    with open(outpath, 'a') as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(art_info)

print("Wrote csv at", outpath)
