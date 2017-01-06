import csv
import re
import sys
import operator
import os
from collections import defaultdict

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

def main():
    csv.field_size_limit(sys.maxsize)

    # Get citations for raw revisions, construct csv with citation bias values
    rawdirpath = '/home/michael/school/cprose_research/wp/wp_articles/extract/extract_articles/ipc_articles_raw/'
    srcpath = '/home/michael/school/cprose_research/wp/wp_articles/article_source_biases.csv'
    
    with open(srcpath, 'w') as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['article_name', 'timestamp', 'editor', 'israeli_citations', 'palestinian_citations', 'neutral_citations'])
        
    for p in os.listdir(rawdirpath):
        art_info = []
        rawpath = os.path.join(rawdirpath, p)
        with open(rawpath, 'r') as rawfile:
            reader = csv.reader(rawfile, delimiter='\t')
            for i, row in enumerate(reader):
                art_name = row[0]
                ts = row[1]
                text = row[2]
                editor = row[3]
                citations = get_citations(text)
                n_isr = 0
                n_pal = 0
                n_neut = 0
                if len(citations) > 0:
                    isrc, palc, neutc = classify_citations(citations)
                    n_isr, n_pal, n_neut = len(isrc), len(palc), len(neutc)
                art_info.append([art_name, ts, editor, n_isr, n_pal, n_neut])
    
        # Write csv
        with open(srcpath, 'a') as out:
            writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(art_info)
        
        print("Wrote citations for {:s}".format(p))
    
    print("Finished writing citation stats at {:s}".format(srcpath))

if __name__ == '__main__':
    main()
