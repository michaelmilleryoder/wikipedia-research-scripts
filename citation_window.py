import csv
import os, re, sys
from collections import Counter
from IPython.core.debugger import Tracer; debug_here = Tracer()
from nltk import word_tokenize

wdwin = 50
cite_diffdir = '/home/michael/school/research/wp/wp_articles/citediffs/'
outpath = '/home/michael/school/research/wp/wp_articles/editor_citewindow_{:d}.csv'.format(wdwin)

# Talk page usernames
csvpath = '/home/michael/school/research/wp/wikipedia/data/talk/ipc_utf8_talkpages.csv'

talk_counter = Counter()

csv.field_size_limit(sys.maxsize)

with open(csvpath, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        talk_counter[row[3]] += 1

talk_eds = set(talk_counter.keys())

# Preprocess text

pats = []

# urls
pats.append(r'https?:\/\/(?:www\.)?[^ ]*')
pats.append(r'[^ ].*?\.html?')

# templates
pats.append(r'\{\{.*?(?:\}\}|$)')
pats.append(r'^.*?\}\}')

# links
pats.append(r'\[\[.*?\|')

# refs
# pats.append(r'<ref name=.*?[>$]')
pats.append(r'<ref name=.*?(?:>|$)')
pat = re.compile('|'.join(pats))

def preprocess(intext):
    outtext = pat.sub('', intext)

    outtext = outtext.replace('[', '')
    outtext = outtext.replace(']', '')
    
    outtext = outtext.replace('{', '')
    outtext = outtext.replace('}', '')
    
    # <ref>
    outtext = outtext.replace('<ref>', '')
    outtext = outtext.replace('</ref>', '')
    
    # titles
    outtext = outtext.replace('=', '')
    
    # categories
    outtext = outtext.replace('Category:', '')
    
    return outtext

# Build editor dictionary with unigrams added
edtext = {ed:'' for ed in talk_eds}

for f in sorted(os.listdir(cite_diffdir)):
    if not f.startswith('.'):
        print(f, end='')
        with open(os.path.join(cite_diffdir,f)) as file:
            r = csv.DictReader(file)
            data = [row for row in r if row['editor'] in talk_eds and row['add/delete']=='add']

    # Iterate thru citediffs, building editor dict

    for i, row in enumerate(data):
        if i % 100 == 0: 
            print('.', end='')
        before = word_tokenize(preprocess(row['before_citation']))
        edtext[row['editor']] += ' '.join(before[max(len(before)*-1, wdwin*-1):])
        after = word_tokenize(preprocess(row['after_citation']))
        edtext[row['editor']] += ' '.join(after[:min(len(after), wdwin)])

    print(' done')

# Write out csv with editor text
with open(outpath, 'w') as f:
    w = csv.writer(f)
    w.writerow(['editor', 'text'])
    for ed in edtext:
        w.writerow([ed, edtext[ed]])

print("Wrote editor text to {:s}".format(outpath))
