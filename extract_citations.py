import csv, sys
import os
import re

citedir = '/home/michael/school/research/wp/wp_articles/citations/'
outpath = '/home/michael/school/research/wp/wp_articles/citations.csv'

citations = []
fps = []
for p in sorted(os.listdir(citedir)):
    fps.append(os.path.join(citedir, p))

for fp in fps:
    print('Parsing citations for {:s}'.format(fp), end='')
    sys.stdout.flush()
    with open(fp) as f:
        rows = [row for row in csv.DictReader(f)]
        
    #for i, row in enumerate(rows[beg:end]):
    for i, row in enumerate(rows):
        if i % 50000 == 0:
            #print(i/1000, 'k')
            print('.', end='')
            sys.stdout.flush()

        # Extract cite book
        if re.search(r'(?:c|C)ite book', row['citation']):
            m_first = re.search(r'(?:c|C)ite book.*first=(?P<first>.*?)\ *?\|', row['citation'])
            m_last = re.search(r'(?:c|C)ite book.*last=(?P<last>.*?)\ *?\|', row['citation'])
            m_author = re.search(r'(?:c|C)ite book.*author=(?P<author>.*?)\ *?\|', row['citation'])
            if m_first and m_last:
                c = ' '.join([m_first.group('first'), m_last.group('last')])
            elif m_first:
                c = m_first.group('first')
            elif m_last:
                c = m_last.group('last')
            elif m_author:
                c = m_author.group('author')
            citations.append(('book', c))

        # Extract cite web
        elif re.search(r'(?:c|C)ite ?web', row['citation']):
            m_auth = re.search(r'(?:c|C)ite ?web.*?(?:author=(?P<author>.*?))(?:\||$)', row['citation'])
            m_url = re.search(r'(?:c|C)ite ?web.*?url=http:\/\/(?:www\.)?(?P<url>.*?)\/', row['citation'])
            if m_auth: 
                citations.append(('web', m_auth.group('author')))
            if m_url: 
                citations.append(('web', m_url.group('url')))

        # Extract cite journal
        elif re.search(r'(?:c|C)ite ?journal', row['citation']):
            m_auth = re.search(r'(?:c|C)ite ?journal.*?(?:author=(?P<author>.*?))(?:\||$)', row['citation'])
            m_journal = re.search(r'(?:c|C)ite ?journal.*?(?:journal=(?P<journal>.*?))(?:\||$)', row['citation'])
            if m_auth: citations.append(('journal', m_auth.group('author')))
            if m_journal: citations.append(('journal', m_journal.group('journal')))

        # Extract cite news
        elif re.search(r'(?:c|C)ite ?news', row['citation']):
            m_auth = re.search(r'(?:c|C)ite ?news\*?(?:author=(?P<author>.*?)\|.*)(?:\||$)', row['citation'])
            m_url = re.search(r'(?:c|C)ite ?news.*?url=http:\/\/(?:www\.)?(?P<url>.*?)\/', row['citation'])
            if m_auth: citations.append(('web', m_auth.group('author')))
            if m_url: citations.append(('web', m_url.group('url')))

        # Extract just url
        elif re.search(r'https?:\/\/(?:www\.)?(?P<url>.*?)\/', row['citation']):
            c = re.search(r'https?:\/\/(?:www\.)?(?P<url>.*?)\/', row['citation']).group('url')
            citations.append(("url", c))

        else: # Default
            m = re.search(r'\ *(?:"|\'\'|\[\[)?(.+?)(?:\.|,|"|“|\'|\||\])', row['citation'])
            if m:
                c = m.group(1)
                citations.append(('unknown', c))
    print("\nFinished parsing")
    print()

    if len(citations) > 1e6:
        with open(outpath, 'a') as f:
            w = csv.writer(f)
            w.writerows(citations)
        print("Wrote rows to {:s}".format(outpath))
        del(citations)
