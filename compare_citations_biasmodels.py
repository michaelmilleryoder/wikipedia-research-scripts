import json, csv, re, pickle, pdb
from collections import defaultdict
from pprint import pprint
from operator import itemgetter
import urllib.request
import urllib.parse

from scipy.stats import pearsonr

# Python 3

""" Creates csv with citation biases and model bias scores
Gets a correlation of revision citation biases and model bias scores
"""

def artlens(latest=False):
	# Create csv for comparing article bias classifications (all articles)
	artlen_csvpath = '/home/michael/school/cprose_research/wp/wikipedia/data/article_lengths.csv'
	pages = [line for line in csv.reader(open(artlen_csvpath,'r'))]
	pages = pages[1:]
	pagelens = {}
	for line in pages:
		if latest:
			pagelens[line[0]] = int(line[-1])
		else:
			pagelens[tuple(line[:3])] = int(line[-1])

	return pagelens


def citebias(lens, latest=False):
	citations_csvpath = '/home/michael/school/cprose_research/wp/wikipedia/data/citations_reverts.csv'
	pages = [line for line in csv.reader(open(citations_csvpath,'r'))]
	citeheader = pages[0]
	pages = pages[1:]
	pagecites = {}
	for line in pages:
		isrcite = int(line[4])
		palcite = int(line[5])
		diff = isrcite - palcite
		total = isrcite+palcite+int(line[6])
		artlen = lens[line[0]] if latest else lens[tuple(line[:3])]
		isrcite_norm = (0 if artlen==0 else float(isrcite)/artlen)
		palcite_norm = (0 if artlen==0 else float(palcite)/artlen)
		diff_norm = (0 if artlen==0 else float(diff)/artlen)
		if latest:
			pagecites[line[0]] = [el for el in line[:2]] + \
					[float(el) for el in line[4:7]] + \
					[diff, total, isrcite_norm, palcite_norm, diff_norm] + \
					[int(el) for el in line[-4:]]
		else:
			pagecites[tuple(line[:3])] = [el for el in line[:2]] + \
					[float(el) for el in line[4:7]] + \
                    [diff, total, isrcite_norm, palcite_norm, diff_norm] + \
					[int(el) for el in line[-4:]]
	
	return pagecites, citeheader


def pwbias(lens, latest=False):
	article_bias_csvpath = '/home/michael/school/cprose_research/wp/wikipedia/data/article_bias_wiki.csv'
	biases = [line for line in csv.reader(open(article_bias_csvpath,'r'))]
	biasheader = biases[0][:4] + biases[0][7:]
	biases = biases[1:]
	pagebiases = {}
	for b in biases:
		artlen = lens[b[0]] if latest else lens[tuple(b[:3])]
		if latest:
			pagebiases[b[0]] = [float(el) for el in b[-6:-2]] + \
					[float(el)/artlen for el in b[-6:-2]]
		else:
			pagebiases[tuple(b[:3])] = [float(el) for el in b[-6:-2]] + \
					[float(el)/artlen for el in b[-6:-2]]
	#article_bias_csvpath = '/home/michael/school/cprose_research/wp/wikipedia/data/ipc_article_biases.csv'
	#biases = [line for line in csv.reader(open(article_bias_csvpath,'r'))]
	#biasheader = biases[0]
	#biases = biases[1:]
	#pagebiases = {}
	#for b in biases:
	#	artlen = lens[b[0]] if latest else lens[tuple(b[:3])]
	#	if latest:
	#		pagebiases[b[0]] = [float(el) for el in b[-6:-2]] + \
	#				[float(el)/artlen for el in b[-6:-2]]
	#	else:
	#		pagebiases[tuple(b[:3])] = [float(el) for el in b[-6:-2]] + \
	#				[float(el)/artlen for el in b[-6:-2]]

	return pagebiases, biasheader

def write_outcsv(csv_outpath, lens, pagecites, citeheader, pagebiases, biasheader):
	with open(csv_outpath, 'w') as out:
		writer = csv.writer(out)
		writer.writerow(citeheader[:2] + citeheader[4:7] + \
				['diff','total_cite','isrcite_norm','palcite_norm','diff_norm'] + \
				citeheader[-4:] + \
                ['art_length'] + biasheader[4:8] + \
				[b+'_norm' for b in biasheader[4:8]])
		for art in sorted(pagecites.keys()):
			if art in pagebiases.keys():
				writer.writerow(
					pagecites[art] + [lens[art]] + list(pagebiases[art]))
	
	print("Wrote output csv to {:s}".format(csv_outpath))

def corrtest(csvpath, stable=True):
	# Article bias latest significance test
	cites = [] # citation diff/artlen
	model = [] # revision bias score/artlen

	print("Loading citation and bias model comparison file...")
	with open(csvpath) as csvfile:
		prev = False # Whether or not previous edit was added to list
		reader = csv.DictReader(csvfile)
		stable_rows = []
		stable_csvpath = csvpath[:-4] + '_stable.csv'
		for i, row in enumerate(reader):
			if i % 50000 == 0:
				print(i/1000, "k")
			rv = True if int(row['revert_md5']) or int(row['revert_comment']) else False
			vandal = True if int(row['vandalism_comment']) else False

			if stable and (rv or vandal): # Take away edit that was reverted
				if prev:
					cites = cites[:-1]
					model = model[:-1]
				prev = False # This edit won't be added to lists

			else:
				if stable:
					stable_rows.append(row)
				if (float(row['israeli_citations']) > 0 or float(row['palestinian_citations']) > 0) and \
						float(row['total_weight_words']) > 100:
					cites.append(float(row['diff_norm']))
					model.append(float(row['bias_norm']))
					prev = True
				else:
					prev = False

	if stable:
		# Write csv with stable csv
		with open(stable_csvpath, 'w') as f:
			writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
			writer.writeheader()
			writer.writerows(stable_rows)
			print("Wrote stable csvpath at", stable_csvpath)

	print("Number of revisions classified: {:d}".format(len(cites)))
	print(pearsonr(cites, model))

def main(path, latest=False, justtest=False, stable=True):
	if latest:
		print("Printing just latest revisions")

	if not justtest:
		print("Getting article lengths...")
		lens = artlens(latest)
		cite, citeheader = citebias(lens, latest)
		print("Getting citation biases...")
		pw, biasheader = pwbias(lens, latest)
		print("Getting polarizing word biases...")

		write_outcsv(path, lens, cite, citeheader, pw, biasheader)

	corrtest(path, stable)

if __name__ == '__main__':
	latest = False
	justtest = True
	path = '/home/michael/school/cprose_research/wp/wikipedia/data/articles_citebias_wiki.csv'
	stable=False
	main(path, latest=latest, justtest=justtest, stable=stable)
