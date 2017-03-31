import csv
import os
import difflib
from collections import Counter
import sys
import re
from urllib.parse import urlsplit
import pdb
csv.field_size_limit(sys.maxsize)

""" 
Calculate diffs between article revisions. 
Output is unordered but not tokenized
"""

""" I/O """
art_dirpath = '/home/michael/school/research/wp/ar/ar_articles'
# I think doesn't include refs and citations and much other Wikipedia markup
diff_dirpath = '/home/michael/school/research/wp/ar/ar_article_diffs'

def preprocess(intext):
    
    text = intext
    
    # Fix strange sentence splitting issue for sentences
    for m in re.findall(r'[a-z]\.[A-Z]', text):
        rep = '. '.join(m.split('.'))
        text = text.replace(m, rep)
        
    # Fix strange sentence splitting issue for sentence + url
    for m in re.findall(r'[a-z]\.http', text):
        rep = '. '.join(m.split('.'))
        text = text.replace(m, rep)
        
    # Add spaces around pipes for image wiki markup
    text = text.replace('|', ' | ')
        
    # Process URLs--just remove http for tokenizer
    for url in re.findall(urlpat, intext):
        try:
            host = urlsplit(url).netloc
        except ValueError as e: 
            continue
        else:
            text = text.replace(url, host)
        
    return text

urlpat = r'https?:\/\/(?:www\.)?[^ ]*'

def main():

    existing = [fname for fname in os.listdir(diff_dirpath)]

    for f in sorted(os.listdir(art_dirpath)):

        if f[:-4]+'_diff.csv' in existing:
            continue

        print("Processing {0}".format(f), end='')
        sys.stdout.flush()
        fpath = os.path.join(art_dirpath, f)
        with open(fpath) as csvfile:
            r = csv.reader(csvfile, delimiter='\t')
            prevc = Counter()
            with open(os.path.join(diff_dirpath, f[:-4]+'_diff.csv'), 'w') as outfile:
                w = csv.writer(outfile)
                w.writerow(['article_name', 'timestamp', 'additions', 'deletions', 'editor', 'edit_comment'])
                for i, row in enumerate(r):
                    if i % 1000 == 0:
                        #print(i/1000, 'k')
                        print('.', end='')
                        sys.stdout.flush()
                    
                    text = preprocess(row[2])
            
                    textc = Counter(text.split())
                    addc = textc - prevc
                    adds = ' '.join(list(addc.elements()))
    #                 adds = ' '.join(word_tokenize(' '.join(list(addc.elements()))))
                    delc = prevc - textc
                    dels = ' '.join(list(delc.elements()))
    #                 dels = ' '.join(word_tokenize(' '.join(list(delc.elements()))))
                    
                    if adds or dels:
                        line = row[:2] + [adds, dels] + row[3:]
                        w.writerow(line)
                        prevc = textc
                
            print("\nWrote diff file for {:s} in {:s}\n".format(f, diff_dirpath))

if __name__ == '__main__':
    main()
