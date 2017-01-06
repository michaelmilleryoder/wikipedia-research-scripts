import csv
import os

# Build blacklist
blacklist_csvpath = '/home/michael/school/research/wp/wikipedia/data/blacklist.csv'
rmpath = '/home/michael/school/research/wp/wp_articles/citations/'
mvpath = '/home/michael/school/research/wp/wp_articles/citations_blacklisted/'

blacklist = []
with open(blacklist_csvpath, 'r') as blfile:
    reader = csv.reader(blfile)
    blacklist = [row[0] for row in reader]
    
# Remove blacklisted articles from a dir (citations)
blacklist_norm = [f.lower().replace(' ', '_') for f in blacklist]

for f in os.listdir(rmpath):
    if f[:-4] in blacklist_norm:
        p = os.path.join(rmpath, f)
        os.renames(p, os.path.join(mvpath, f))
        print("Moved file {:s}".format(f))
