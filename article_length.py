import csv
import sys
import os
csv.field_size_limit(sys.maxsize)

rawdirpath = '/home/michael/school/cprose_research/wp/wp_articles/extract/extract_articles/ipc_articles_raw/'
outpath = '/home/michael/school/cprose_research/wp/wp_articles/article_lengths.csv'

data = []

for p in os.listdir(rawdirpath):
    rawpath = os.path.join(rawdirpath, p)
    with open(rawpath, 'r') as rawfile:
        reader = csv.reader(rawfile, delimiter='\t')
        art_info = [row[:2] + row[3:5] + [len(row[2].split(' '))] for row in reader]
    data.append(art_info)
    print(p)

# Write csv
with open(outpath, 'w') as out:
    writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['article_name', 'timestamp', 'editor', 'edit_comment', 'article_length'])
    writer.writerows(sum(data, []))
    print("Wrote file at", outpath)

