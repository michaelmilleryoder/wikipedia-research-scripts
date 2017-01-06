import csv
import os
import re, sys
from collections import defaultdict, Counter
from operator import itemgetter
import pandas as pd
import math
import urllib.request
import pdb

OUTPATH = '/home/michael/school/research/wp/wp_articles/editor_citation_svdfeature_citetype.txt'

# Initialize for aggregating citations for top sources (also a script)
ec = defaultdict(lambda: defaultdict(Counter)) # ec[name][citation][year]
srcs = set()
citedir = '/home/michael/school/research/wp/wp_articles/citediffs/'
ecmatpath = '/home/michael/school/research/wp/wp_articles/editor_citations_matrix.csv'
fps = []
for p in sorted(os.listdir(citedir)):
    if not p.startswith('.'): fps.append(os.path.join(citedir, p))

# Aggregate citations from citediffs
print('Aggregating citations...', end='')
sys.stdout.flush()
for fp in fps:
    with open(fp) as f:
        rows = [row for row in csv.DictReader(f)]
        
    for i, row in enumerate(rows):
        feat = row['citation_type']
        #feat = row['timestamp'][:4]
        if row['add/delete'] == 'add':
            if row['vandalism_comment'] == '0':
                ec[row['editor']][row['citation']][feat] += 1
                srcs.add(row['citation'])
        elif row['add/delete'] == 'delete':
            if row['vandalism_comment'] == '0':
                ec[row['editor']][row['citation']][feat] -= 1
print('done')
    
srcs = sorted(list(srcs))

# Get 100 most frequent citations
freqpath = '/home/michael/school/research/wp/wikipedia/data/citations_annotated.csv'
freq = pd.read_csv(freqpath)
freq100 = freq.loc[:99,:]

freqsrcs = {}
for _,row in freq100.iterrows():
    if isinstance(row['variants'], str):
        freqsrcs[row['name']] = [row['citation']]+ [var[1:-1] for var in row['variants'][1:-1].split(', ')]
    else:
        freqsrcs[row['name']] = [row['citation']]
len(freqsrcs)

# Get bot list
print('Getting bot list...', end='')
sys.stdout.flush()
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
    cnum = re.search(r'cmcontinue":"([^"]*)"', page)
    if cnum:
        cnum = cnum.group(1)
#         print(cnum)
    else:
        done = True

# Add other bots
other_bots = ['SmackBot']
bots += other_bots
print('done')

# Print out editor citation matrix
print("Creating editor-citation matrix...", end='')
sys.stdout.flush()
ec100 = {}
variants = [x for sublist in [val for val in freqsrcs.values()] for x in sublist]

# Reconstruct a combined one
for ed in [e for e in ec if e not in bots and e != '']:
    for src in ec[ed]:
        if src in variants:
            # Find which is name
            selected_variants = [val for val in freqsrcs.values() if src in val][0]
            name = list(freqsrcs.keys())[list(freqsrcs.values()).index(selected_variants)]
            if not ed in ec100: ec100[ed] = {}
            if not name in ec100[ed]: ec100[ed][name] = {}
            for feat in ec[ed][src]:
                if not feat in ec100[ed][name]:
                    ec100[ed][name][feat] = ec[ed][src][feat]
                else:
                    ec100[ed][name][feat] += ec[ed][src][feat]

ecmat100 = pd.DataFrame(ec100)

# Write editor-citation matrix
ecmat100.to_csv(ecmatpath, na_rep=0)
print('done')

# Print input for SVDFeature--sources and editors sorted alphabetically
ratings = []
for i, ed in enumerate(sorted(ec100.keys())):
    for j, src in enumerate(sorted(freqsrcs.keys())):
        if src in ec100[ed]:
            for k, feat in enumerate(sorted(ec100[ed][src])):

                # Citation feature
                rating = [ec100[ed][src][feat], 0, 1, 2, 
                        "{:s}:1".format(feat), 
                        "{:d}:{:d}".format(i+1,1), 
                        "{:d}:{:d}".format(j+1,1)]

                # Global feature
                rating = [ec100[ed][src][feat], 1, 1, 1, 
                        "{:s}:1".format(feat), 
                        "{:d}:{:d}".format(i+1,1), 
                        "{:d}:{:d}".format(j+1,1)]

                ratings.append(rating)
    
with open(OUTPATH, 'w') as f:
    w = csv.writer(f, delimiter=' ')
    for r in ratings:
        w.writerow(r)

print("Wrote SVDFeature input to {:s}".format(OUTPATH))
