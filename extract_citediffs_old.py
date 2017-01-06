import csv
import os
import re, sys
from collections import defaultdict, Counter
import pdb

# Aggregate added, deleted citations for each revision
citedir = '/home/michael/school/research/wp/wp_articles/citations/'
cite_diffdir = '/home/michael/school/research/wp/wp_articles/citediffs/'
infopath = '/home/michael/school/research/wp/wikipedia/data/citations_reverts.csv'

citations = Counter()
adds = Counter()
dels = Counter()
fps = []

urlpat = r'https?:\/\/(?:www\.)?(?P<url>[^\/\] ]*)' 

for p in sorted(os.listdir(citedir)):
    if not p.startswith('.'): fps.append(os.path.join(citedir, p))

# Load article revision metadata
print('Loading metadata...')
with open(infopath) as f:
    r = csv.DictReader(f)
    info = {(row['article_name'], row['timestamp']): row for row in r}

# Parse added, deleted citations for each revision
#for fp in fps[1:2]:
for fp in fps:
    print('Parsing citations in {:s}'.format(os.path.split(fp)[1]), end='')
    sys.stdout.flush()

    # Write header
    with open(os.path.join(cite_diffdir, os.path.split(fp)[1][:-14]) + '_citediff.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['article_name', 'timestamp', 'editor', 'add/delete', 'citation_type', 
                'citation', 'citation_text', 'before_citation', 'after_citation', 'edit_comment', 
               'edit_length', 'revert_md5', 'revert_comment', 'vandalism_comment'])

        prev_cites = Counter()
        cites = Counter()

        with open(fp) as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i == 0:
                    prev_ts = row['timestamp']
                if i % 50000 == 0:
                    print('.', end='')
                    sys.stdout.flush()
                    
                ts = row['timestamp']
                ri = info[(row['article_name'], ts)]
                
                # Update
                if ts != prev_ts: 
                    d = prev_cites-cites
                    a = cites-prev_cites
                    for c in a:
                        w.writerow([row['article_name'], ts, ri['editor'], 'add', c[0], c[1], ri['edit_comment'], ri['edit_length'],
                                       ri['revert_md5'], ri['revert_comment'], ri['vandalism_comment']])
                    for c in d:
                        w.writerow([row['article_name'], ts, ri['editor'], 'delete', c[0], c[1], ri['edit_comment'], ri['edit_length'],
                                       ri['revert_md5'], ri['revert_comment'], ri['vandalism_comment']])
                    
                    prev_cites = cites
                    cites = Counter()
                    prev_ts = ts

                # Extract cite book
                if re.search(r'(?:c|C)ite book', row['citation']):
                    m_first = re.search(r'(?:c|C)ite book.*first=(?P<first>.*?)\ *?\|', row['citation'])
                    m_last = re.search(r'(?:c|C)ite book.*last=(?P<last>.*?)\ *?\|', row['citation'])
                    m_author = re.search(r'(?:c|C)ite book.*author=(?P<author>.*?)\ *?\|', row['citation'])
                    m_title = re.search(r'(?:c|C)ite book.*title=(?P<title>.*?)\ *?\|', row['citation'])
                    if m_first and m_last:
                        cites[('book',' '.join([m_first.group('first'), m_last.group('last')]))] += 1
                    elif m_last:
                        cites[('book', m_last.group('last'))] += 1
                    elif m_author:
                        cites[('book', m_author.group('author'))] += 1
                    elif m_title:
                        cites[('book', m_title.group('title'))] += 1

                # Extract cite web
                elif re.search(r'(?:c|C)ite ?web', row['citation']):
                    m_auth = re.search(r'(?:c|C)ite ?web.*?(?:author=(?P<author>.*?))(?:\||$)', row['citation'])
                    m_url = re.search(r'(?:c|C)ite ?web.*?url='+urlpat, row['citation'])
                    if m_auth: 
                        cites[('web', m_auth.group('author'))] += 1
                        #cites[m_auth.group('author')] += 1
                    if m_url: 
                        cites[('web', m_url.group('url'))] += 1
                        #cites[m_url.group('url')] += 1

                # Extract cite journal
                elif re.search(r'(?:c|C)ite ?journal', row['citation']):
                    m_auth = re.search(r'(?:c|C)ite ?journal.*?(?:author=(?P<author>.*?))(?:\||$)', row['citation'])
                    m_journal = re.search(r'(?:c|C)ite ?journal.*?(?:journal=(?P<journal>.*?))(?:\||$)', row['citation'])
                    if m_auth: 
                        cites[('journal', m_auth.group('author'))] += 1
                        #cites[m_auth.group('author')] += 1
                    if m_journal: 
                        cites[('journal', m_journal.group('journal'))] += 1
                        #cites[m_journal.group('journal')] += 1

                # Extract cite news
                elif re.search(r'(?:c|C)ite ?news', row['citation']):
                    m_auth = re.search(r'(?:c|C)ite ?news\*?(?:author=(?P<author>.*?)\|.*)(?:\||$)', row['citation'])
                    m_url = re.search(r'(?:c|C)ite ?news.*?url='+urlpat, row['citation'])
                    if m_auth: 
                        cites[('news', m_auth.group('author'))] += 1
                        #cites[m_auth.group('author')] += 1
                    if m_url: 
                        cites[('news', m_url.group('url'))] += 1
                        #cites[m_url.group('url')] += 1

                # Extract just url
                elif re.search(urlpat, row['citation']):
                    c = re.search(urlpat, row['citation']).group('url')
                    cites['web', c] += 1
                    #cites[c] += 1
                    
                # Extract harvnb
                elif re.search(r'[Hh]arvnb\|(.*?)\|', row['citation']):
                    c = re.search(r'[Hh]arvnb\|(.*?)\|', row['citation']).group(1)
                    cites[('other', c)] += 1
                    #cites[c] += 1

                # Extract harvtxt
                elif re.search(r'[Hh]arvtxt\|(.*?)\|', row['citation']):
                    c = re.search(r'[Hh]arvtxt\|(.*?)\|', row['citation']).group(1)
                    cites[('other', c)] += 1
                    #cites[c] += 1

                else: # Default
                    m = re.search(r'\ *(?:"|\'\'|\[\[)?(.+?)(?:\.|,|"|“|\'|\||\])', row['citation'])
                    if m:
                        c = m.group(1)
                        cites[('other', c)] += 1
                        #cites[c] += 1
                    #else:
                    #    print("no parse")
                
    print("finished")
