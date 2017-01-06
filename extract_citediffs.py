import csv
import os
import re, sys
from collections import defaultdict, Counter
import pdb

# Aggregate added, deleted citations for each revision
citedir = '/home/michael/school/research/wp/wp_articles/citations/'
cite_diffdir = '/home/michael/school/research/wp/wp_articles/citediffs/'
revdir = '/home/michael/school/research/wp/wp_articles/extract/extract_articles/ipc_articles_raw'
infopath = '/home/michael/school/research/wp/wikipedia/data/citations_reverts.csv'
charwin = 1000

citations = Counter()
adds = Counter()
dels = Counter()
fps = []

urlpat = r'https?:\/\/(?:www\.)?(?P<url>[^\/\] ]*)' 

csv.field_size_limit(sys.maxsize)

# Process text
def preprocess(intext):
    ''' Takes a long time'''

    # urls
    outtext = re.sub(r'https?:\/\/(?:www\.)?[^ ]*', '', intext)
    outtext = re.sub(r'[^ ].*?\.html?', '', outtext)
    
    # templates
    outtext = re.sub(r'\{\{.*?(?:\}\}|$)','', outtext)
    outtext = re.sub(r'^.*?\}\}','', outtext)
    
    # links
    outtext = re.sub(r'\[\[.*?\|', '', outtext)
    outtext = outtext.replace('[', '')
    outtext = outtext.replace(']', '')
    
    # <ref>
    outtext = outtext.replace('<ref>', '')
    outtext = outtext.replace('</ref>', '')
    
    # titles
    outtext = outtext.replace('=', '')
    
    # categories
    outtext = outtext.replace('Category:', '')
    
    return outtext

# Extract citations from raw citation text
def extract_citation(intext):
    ''' Returns tuple (type, citation, citation_text) '''
    citeinfo = 'other', '', intext

    # Extract cite book
    if re.search(r'(?:c|C)ite book', intext):
        m_first = re.search(r'(?:c|C)ite book.*first=(?P<first>.*?)\ *?\|', intext)
        m_last = re.search(r'(?:c|C)ite book.*last=(?P<last>.*?)\ *?\|', intext)
        m_author = re.search(r'(?:c|C)ite book.*author=(?P<author>.*?)\ *?\|', intext)
        m_title = re.search(r'(?:c|C)ite book.*title=(?P<title>.*?)\ *?\|', intext)
        if m_first and m_last:
            citeinfo = 'book',' '.join([m_first.group('first'), m_last.group('last')]), intext
        elif m_last:
            citeinfo = 'book', m_last.group('last'), intext
        elif m_author:
            citeinfo = 'book', m_author.group('author'), intext
        elif m_title:
            citeinfo = 'book', m_title.group('title'), intext

    # Extract cite web
    elif re.search(r'(?:c|C)ite ?web', intext):
        m_auth = re.search(r'(?:c|C)ite ?web.*?(?:author=(?P<author>.*?))(?:\||$)', intext)
        m_url = re.search(r'(?:c|C)ite ?web.*?url='+urlpat, intext)
        if m_auth: 
            citeinfo = 'web', m_auth.group('author'), intext
        if m_url: 
            citeinfo = 'web', m_url.group('url'), intext

    # Extract cite journal
    elif re.search(r'(?:c|C)ite ?journal', intext):
        m_auth = re.search(r'(?:c|C)ite ?journal.*?(?:author=(?P<author>.*?))(?:\||$)', intext)
        m_journal = re.search(r'(?:c|C)ite ?journal.*?(?:journal=(?P<journal>.*?))(?:\||$)', intext)
        if m_auth: 
            citeinfo = 'journal', m_auth.group('author'), intext
        if m_journal: 
            citeinfo = 'journal', m_journal.group('journal'), intext

    # Extract cite news
    elif re.search(r'(?:c|C)ite ?news', intext):
        m_auth = re.search(r'(?:c|C)ite ?news\*?(?:author=(?P<author>.*?)\|.*)(?:\||$)', intext)
        m_url = re.search(r'(?:c|C)ite ?news.*?url='+urlpat, intext)
        if m_auth: 
            citeinfo = 'news', m_auth.group('author'), intext
        if m_url: 
            citeinfo = 'news', m_url.group('url'), intext

    # Extract just url
    elif re.search(urlpat, intext):
        c = re.search(urlpat, intext).group('url')
        citeinfo = 'web', c, intext
        
    # Extract harvnb
    elif re.search(r'[Hh]arvnb\|(.*?)\|', intext):
        c = re.search(r'[Hh]arvnb\|(.*?)\|', intext).group(1)
        citeinfo = 'other', c, intext

    # Extract harvtxt
    elif re.search(r'[Hh]arvtxt\|(.*?)\|', intext):
        c = re.search(r'[Hh]arvtxt\|(.*?)\|', intext).group(1)
        citeinfo = 'other', c, intext

    else : # Default
        m = re.search(r'\ *(?:"|\'\'|\[\[)?(.+?)(?:\.|,|"|“|\'|\||\])', intext)
        if m:
            c = m.group(1)
            citeinfo = 'other', c, intext
    
    return citeinfo

for p in sorted(os.listdir(citedir)):
    root = p.split('_citations.csv')[0]
    if not p.startswith('.') and root+'.tsv' in os.listdir(revdir): 
        fps.append(os.path.join(citedir, p))

# Load article revision metadata
print('Loading metadata...')
with open(infopath) as f:
    r = csv.DictReader(f)
    info = {(row['article_name'], row['timestamp']): row for row in r}

# Parse added, deleted citations for each revision
#for fp in fps[1:2]:
for fp in fps:
    print('Parsing citations in {:s}'.format(os.path.split(fp)[1]), end='')
    root = os.path.split(fp)[1].split('_citations.csv')[0]
    sys.stdout.flush()

    # Load revision text
    with open(os.path.join(revdir, root+'.tsv')) as f:
        r = csv.reader(f, delimiter='\t')
        revs = {row[1]: row[2] for row in r} # timestamp: text

    # Write header
    with open(os.path.join(cite_diffdir, os.path.split(fp)[1][:-14]) + '_citediff.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['article_name', 'timestamp', 'editor', 'add/delete', 'citation_type', 
                'citation', 'citation_text', 'before_citation', 'after_citation', 'edit_comment', 
               'edit_length', 'revert_md5', 'revert_comment', 'vandalism_comment'])

        prev_cites = Counter()
        cites = Counter() # citation info
        cite_text = Counter() # citation text

        with open(fp) as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i == 0:
                    prev_ts = row['timestamp']
                if i % 50000 == 0:
                #if i % 10000 == 0:
                    print('.', end='')
                    sys.stdout.flush()
                    
                ts = row['timestamp']
                ri = info[(row['article_name'], ts)]
                
                # Update
                if ts != prev_ts: 
                    #d = prev_cites-cites
                    #a = cites-prev_cites
                    d = prev_cites-cite_text
                    a = cite_text-prev_cites

                    for citetext in a:

                        c = extract_citation(citetext) 

                        # Get citation context
                        text = revs[ts]        
                        cidx = text.rfind(citetext) 
                        after_idx = cidx+len(c)+1

                        #before = preprocess(text[max(cidx-charwin,0):cidx])
                        #after = preprocess(text[after_idx:min(after_idx+charwin,len(text))])
                        before = text[max(cidx-charwin,0):cidx]
                        after = text[after_idx:min(after_idx+charwin,len(text))]

                        w.writerow([row['article_name'], ts, ri['editor'], 'add', c[0], c[1], c[2], before, after, ri['edit_comment'], ri['edit_length'],
                                       ri['revert_md5'], ri['revert_comment'], ri['vandalism_comment']])

                    for citetext in d:

                        c = extract_citation(citetext) 

                        w.writerow([row['article_name'], ts, ri['editor'], 'delete', c[0], c[1], c[2], '', '', ri['edit_comment'], ri['edit_length'],
                                       ri['revert_md5'], ri['revert_comment'], ri['vandalism_comment']])
                    
                    prev_cites = cite_text
                    #prev_cites = cites
                    #cites = Counter()
                    cite_text = Counter()
                    prev_ts = ts

                # Add citation to timestamp
                cite_text[row['citation']] += 1
                
    print(" finished")
