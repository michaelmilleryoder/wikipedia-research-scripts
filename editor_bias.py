import csv, json, urllib, re
from pprint import pprint
from scipy.stats import pearsonr
from operator import itemgetter
from collections import defaultdict
import pandas as pd
import pdb
import numpy as np

def bots():
	# Get bot list
	bots = []
	cnum = ""
	done = False

	# Get bot category page
	category_base_url = "https://en.wikipedia.org/w/api.php?" + \
		"action=query&" +  \
		"list=categorymembers&" + \
		"format=json&" + \
		"cmtitle=Category%3A{:s}&" +  \
		"cmprop=title&" + \
		"cmlimit=500" + \
		"&cmcontinue={:s}"
		
	while not done:
		category_url = category_base_url.format(urllib.parse.quote("All_Wikipedia_bots"), cnum)
		#print(category_url)

		page =  urllib.request.urlopen(category_url).read().decode('utf-8')
		#print(page)

		# Extract bot names
		botsp = re.findall(r'"User\:(?P<username>[\w\ /\.]*)"\}', page)
		botsp = [re.sub(r'/.*', '', ed) for ed in botsp]
		#print(botsp[:10])
		bots += botsp

		# Get continue number
		#cnum = re.search(r'cmcontinue":"(?P<cnum>[^"]*)"', page)
		cnum = re.search(r'cmcontinue":"([^"]*)"', page)
		if cnum:
			cnum = cnum.group(1)
			#print(cnum)
		else:
			done = True

	# Add other bots
	other_bots = ['SmackBot', 'Tawkerbot2']
	bots += other_bots

	return bots

def modelbias(modelfile, edit_threshold, savejson=None):
	""" Returns model user scores
	"""

	print("Getting model user scores...")
	b = bots()

	# Get user edit biases
	sum_biases = defaultdict(float)
	avg_biases = defaultdict(float)
	editcounts = defaultdict(int)
	with open(modelfile) as csvfile:
		for row in csv.DictReader(csvfile):
			name = row['Username']
			if name != '' and name not in b:
				editcounts[name] += 1
				if editcounts[name] >= edit_threshold:
					sum_biases[row['Username']] += float(row['User avg edit bias'])
					avg_biases[row['Username']] = float(row['User avg edit bias'])

	# Save json of user biases
	if savejson:
		with open(savejson + '_sum_' + str(edit_threshold) + '.json', 'w') as out:
			json.dump(sum_biases, out)
			print("Saved sum bias json file to", savejson)
		with open(savejson + '_avg_' + str(edit_threshold) + '.json', 'w') as out:
			json.dump(avg_biases, out)
			print("Saved avg bias json file to", savejson)

	return sum_biases, avg_biases

def create_editor_list(modelfile, edit_threshold, nquantiles, savejson=None):
	print("Creating editor list...")

	sum_biases, _ = modelbias(modelfile, edit_threshold, savejson)

	# Open JSON files of editor citation bias lists
	filepath = '/home/michael/school/cprose_research/wp/wikipedia/data/editor_citationbias_novandal.json'
	with open(filepath) as f:
		novandal = json.load(f)

	# combined sums
	combinedsum = {}
	for line in novandal:
		name = line[0]
		if name in sum_biases.keys():
			bias = sum_biases[name]
			combinedsum[name] = {'sumbias': bias, 'citebias': line[1][1]}

	ab = []
	cb = []
	names = []
	for name in combinedsum:
		if combinedsum[name]['citebias'] != 0.0:
			names.append(name)
			ab.append(combinedsum[name]['sumbias'])
			cb.append(combinedsum[name]['citebias'])
	
	data = pd.DataFrame([z for z in zip(names, cb, ab)], columns=['name', 'citation_bias', 'sum_bias'])

	if nquantiles >= 2:
		# Get quantiles
		data['sum_bias_category'] = pd.qcut(data['sum_bias'], nquantiles, labels=range(nquantiles))

		# Get quantiles for citation bias
		data['citation_bias_category'] = pd.qcut(data['citation_bias'], nquantiles, labels=range(nquantiles))

		data['citation_bias_category'] = data['citation_bias_category'].cat.remove_categories(range(1, nquantiles-1)) # neutral is nan
		data['citation_bias_category'] = data['citation_bias_category'].cat.rename_categories(['pal', 'israeli'])

		data['sum_bias_category'] = data['sum_bias_category'].cat.remove_categories(range(1, nquantiles-1)) # neutral is nan
		data['sum_bias_category'] = data['sum_bias_category'].cat.rename_categories(['pal', 'israeli'])

	else: # Split based on 0
		cite_cats = []
		for c in cb:
			cite_cats.append('isr' if c>0 else 'pal')
		sum_cats = []
		for s in ab:
			sum_cats.append('isr' if s>0 else 'pal')
		data['sum_bias_category'] = sum_cats
		data['citation_bias_category'] = cite_cats

	return data

def printstats(data):

	# Count accuracy
	ncorrect = 0

	for row in data.values:
		a = row[-2]
		b = row[-1]
		if row[-2] == row[-1]:
			ncorrect += 1
		elif isinstance(a, float) and isinstance(b, float):
			if np.isnan(row[-2]) and np.isnan(row[-1]):
				ncorrect += 1

	print('Accuracy', ncorrect, '/', len(data), ncorrect/len(data))
	print('Correlation', pearsonr(data['citation_bias'].values, data['sum_bias'].values))

def writecsv(data, nquantiles, path='/home/michael/school/cprose_research/wp/wikipedia/data/editor_categories.csv'):
	data.to_csv(path[:-4] + '_{:d}q.csv'.format(nquantiles))
	print("Wrote csv to", path, nquantiles)

def readcsv(path='/home/michael/school/cprose_research/wp/wikipedia/data/editor_categories.csv'):
	return pd.read_csv(path)

def main():
	edit_threshold = 15
	nquantiles = 0
	modelfile = '/home/michael/school/cprose_research/wp/wikipedia/data/IPC_userBiases_wiki.csv'
	data = create_editor_list(modelfile, edit_threshold, nquantiles)
	#data = readcsv()
	printstats(data)
	writecsv(data, nquantiles, '/home/michael/school/cprose_research/wp/wikipedia/data/editor_categories_wiki.csv')

	#modelbias(edit_threshold, savejson='/home/michael/school/cprose_research/wp/wikipedia/data/editor_bias')

if __name__ == '__main__':
	main()
