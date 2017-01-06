import csv
import os
import re, sys
from collections import Counter

""" Aggregates citations into a top list of citations by additions and deletions """

# Initialize for aggregating citations
citations = Counter()
adds = Counter()
dels = Counter()
citedir = '/home/michael/school/research/wp/wp_articles/citediffs/'
outpath = '/home/michael/school/research/wp/wp_articles/citations_add_del.csv'
fps = []
for p in sorted(os.listdir(citedir)):
    if not p.startswith('.'): fps.append(os.path.join(citedir, p))

# Aggregate citations from citediffs
print("Aggregating citations")
for fp in fps:
    with open(fp) as f:
        rows = [row for row in csv.DictReader(f)]
        
    for i, row in enumerate(rows):
        if row['add/delete'] == 'add':
            if row['vandalism_comment'] == '0':
                adds[row['citation']] += 1
                citations[row['citation']] += 1
        elif row['add/delete'] == 'delete':
            if row['vandalism_comment'] == '0':
                dels[row['citation']] += 1
                citations[row['citation']] += 1
    
# Print out top citations: adds, dels, citations
with open(outpath, 'w') as f:
    w = csv.writer(f)
    w.writerow(['citation', 'additions', 'deletions', 'total'])
    for c in citations.most_common():
        name = c[0]
        w.writerow([name, adds[name], dels.get(name, 0), citations[name]])

print("Wrote out top citations at {:s}".format(outpath))
