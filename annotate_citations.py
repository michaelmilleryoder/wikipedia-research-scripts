import csv
import os
import re, sys
from collections import defaultdict, Counter
from operator import itemgetter
import pandas as pd

ANNOTATED_CSVPATH = '/home/michael/school/research/wp/wp_articles/citations_annotated.csv'

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

# Annotate add_del figures
raw_csvpath = '/home/michael/school/research/wp/wp_articles/citations_add_del.csv'
raw = pd.read_csv(raw_csvpath)

variants = [x for sublist in [val for val in freqsrcs.values()] for x in sublist]

# Reconstruct a combined one
counts = {}
for i, row in raw.iterrows():
    if row['citation'] in variants:
        # Find which is name
        selected_variants = [val for val in freqsrcs.values() if row['citation'] in val][0]
        name = list(freqsrcs.keys())[list(freqsrcs.values()).index(selected_variants)]
        #print(row['citation'], name)
        if name in counts:
            counts[name]['additions'] += row['additions']
            counts[name]['deletions'] += row['deletions']
            counts[name]['total'] += row['total']
            if 'variants' in counts[name]:
                counts[name]['variants'].append(row['citation'])
            else:
                counts[name]['variants'] = [row['citation']]
                
        else:
            counts[name] = {}
            counts[name]['citation'] = row['citation']
            counts[name]['additions'] = row['additions']
            counts[name]['deletions'] = row['deletions']
            counts[name]['total'] = row['total']
            
    elif row['citation'] in counts: # name of the source
        counts[row['citation']]['additions'] += row['additions']
        counts[row['citation']]['deletions'] += row['deletions']
        counts[row['citation']]['total'] += row['total']
        if 'variants' in counts[row['citation']]:
            counts[row['citation']]['variants'].append(row['citation'])
        else:
            counts[row['citation']]['variants'] = [row['citation']]
        
    else:
        counts[row['citation']] = {}
        counts[row['citation']]['citation'] = row['citation']
        counts[row['citation']]['additions'] = row['additions']
        counts[row['citation']]['deletions'] = row['deletions']
        counts[row['citation']]['total'] = row['total']

annotated = pd.DataFrame(counts).transpose()
annotated['name'] = annotated.index
annotated = annotated.sort(columns=['total'], ascending=False)
annotated = annotated[['name', 'total', 'additions', 'deletions', 'citation', 'variants']]

# Write new annotated csv
annotated.to_csv(ANNOTATED_CSVPATH, index=False)
print("Wrote annotated source list to {:s}".format(ANNOTATED_CSVPATH))
