import csv
import re
import sys
import operator
import os
from collections import defaultdict, Counter
import urllib.request

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

def changes(prevlist, curlist):
    """ Returns changes in list counts """
    changes = defaultdict(int)

    c0 = Counter(prevlist)
    c1 = Counter(curlist)
    rms = c0-c1
    adds = c1-c0

    for el in rms:
        changes[el] -= rms[el]
    for el in adds:
        changes[el] += adds[el]

    return changes

# Get bot list
bots = []
cnum = ""
done = False

# Get bot category page
category_base_url = "https://en.wikipedia.org/w/api.php?" +     "action=query&" +      "list=categorymembers&" +     "format=json&" +     "cmtitle=Category%3A{:s}&" +      "cmprop=title&" +     "cmlimit=500" +     "&cmcontinue={:s}"
    
while not done:
    category_url = category_base_url.format(urllib.parse.quote("All_Wikipedia_bots"), cnum)
    #print(category_url)

    page =  urllib.request.urlopen(category_url).read().decode('utf-8')
    #print(page)

    # Extract bot names
    botsp = re.findall(r'"User\:(?P<username>[\w\ /\.]*)"\}', page)
    botsp = [re.sub(r'/.*', '', ed) for ed in botsp]
    #print(botsp[:10])
    bots += botsp
#     print(len(bots))

    # Get continue number
    #cnum = re.search(r'cmcontinue":"(?P<cnum>[^"]*)"', page)
    cnum = re.search(r'cmcontinue":"([^"]*)"', page)
    if cnum:
        cnum = cnum.group(1)
#         print(cnum)
    else:
        done = True

# Add other bots
other_bots = ['SmackBot']
bots += other_bots


# Build revert dictionary

csvpath = '/home/michael/school/cprose_research/wp/wikipedia/data/citations_reverts.csv'

reverts = defaultdict(bool)
#ec = defaultdict(dict) # format editor: {citation: count}

with open(csvpath, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for i, row in enumerate(reader):
        #if i % 100000 == 0:
            #print(i/1000, 'k')
        
        artname = row[0]
        ts = row[1]
        editor = row[2]
        mdf5 = int(row[8])
        rvcomment = int(row[9])
        rv = max(mdf5, rvcomment)
        rvv = int(row[10])

        if rv or rvv:
            reverts[(artname, ts, editor)] = True
            

# Get citation diffs for editors
rawdirpath = '/home/michael/school/cprose_research/wp/wp_articles/extract/extract_articles/ipc_articles_raw/'
ec = defaultdict(dict) # format {editor: {citation: count}}

for p in sorted(os.listdir(rawdirpath)):
    artlines = []
    cite_info = defaultdict(int) # dict for citations for each article (citation, art_name): count
    rawpath = os.path.join(rawdirpath, p)
    prevs = list()
    with open(rawpath, 'r') as rawfile:
        reader = csv.reader(rawfile, delimiter='\t')
        for i, row in enumerate(reader): # for every revision
            art_name = row[0]
            text = row[2]
            ts = row[1]
            editor = row[3]
            citations = get_citations(text)
            if not reverts[(art_name, ts, editor)] and not editor in bots:
                isrc, palc, neutc = classify_citations(citations)
                if len(isrc) > 0 or len(palc) > 0:
                    ch = changes([p[0] for p in prevs], [d[0] for d in isrc+palc])
                    if ch:
#                         debug_here()
                        for c in ch:
                            if c in ec[editor].keys():
                                ec[editor][c] += ch[c] # Change count for that editor, for that citation
                            else:
                                ec[editor][c] = ch[c] # Change count for that editor, for that citation
            prevs = isrc+palc
                        
    print("Finished with {:s}".format(p))                        
    

# Make editor matrix from input csv
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

srcs = sorted([el[0] for el in isr_srcs+pal_srcs])
outrows = [[""] + srcs]

for a in sorted(ec.keys()):
    outrow = [a]
    for s in srcs:
        if s in ec[a].keys():
            outrow.append(ec[a][s])
        else:
            outrow.append(0)
    outrows.append(outrow)
    
csvpath = '/home/michael/Desktop/editor_citation_matrix.csv'
with open(csvpath, 'w') as f:
    w = csv.writer(f)
    w.writerows(outrows)

print("Saved editor matrix at {:s}".format(csvpath))
