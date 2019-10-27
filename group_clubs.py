# Cal Hacks 6.0
# obtain URLs for all organizations
import urllib.request
import sys, os, csv
from googlesearch import search
# urls = search('site:callink.berkeley.edu inurl:organization/', stop=20,num=10,pause=2.0)

# open URLs and read in descriptions

def remove_html_tags(text):
	"""Remove html tags from a string"""
	import re
	clean = re.compile('<.*?>')
	return re.sub(clean, ' ', text)

club_info = {}
pages = []
# for url in urls:
# for url in open('club_urls.txt'):
# 	try:
# 		page = str(urllib.request.urlopen(url).read())
# 	except:
# 		sys.stderr.write("Can't open {}\n".format(url))
# 		continue
# 	pages += [page]
# 	name = page[page.find('"name"')+8:]
# 	name = name[:name.find('",')]
# 	name = name.replace('\\', '')

# 	description = page[page.find('"description"')+15:]
# 	if 'description' not in page:
# 		continue
# 	if '",' not in description:
# 		end = 500
# 	else:
# 		end = description.find('",')
# 	description = remove_html_tags(description[:end])
# 	description = description.replace('\\r\\','')
# 	description = description.replace('\\n','')
# 	description = description.replace('\\','')
# 	description = description.replace('&nbsp;','')	
# 	club_info[name] = description

# with open('club_html_pages.txt', 'wt') as outfile:
# 	writer = csv.writer(outfile, delimiter='\t', lineterminator=os.linesep)
# 	for p in pages:
# 		writer.writerow([p])

# with open('club_descriptions.txt', 'wt') as outfile:
# 	writer = csv.writer(outfile, delimiter='\t', lineterminator=os.linesep)
# 	for name in club_info:
# 		writer.writerow([name, club_info[name]])

for line in open('club_descriptions.txt'):
	name, description = line.rstrip().split('\t')
	club_info[name] = description

# calculating similarity of club descriptions and then clustering clubs by their similarities
# these libraries are pip installable 
from textblob import TextBlob
from sklearn.cluster import SpectralClustering
# from gensim import corpora, models, similarities
# from collections import defaultdict

def normalize(vec):
	val = max(vec) or 1
	return [v/val for v in vec]


stoplist = set('for a of the and to in'.split())

club_names = sorted(club_info.keys())
club_summary = {}
club_sentiment = {}
club_lsi = {}
for club in club_names:
	club_description = club_info[club]
	# club_lsi = corpora.MmCorpus
	# cleaned_description = [word for word in club_description.lower().split() if word not in stoplist]
	# frequency = defaultdict(int)
	# for text in cleaned_description:
	# 	frequency[text] += 1
	# print(frequency)
	# dictionary = corpora.Dictionary(frequency)
	# club_lsi = models.LsiModel([dictionary.doc2bow(text) for text in cleaned_description], id2word=dictionary, num_topics=30)

	club_description_textblob = TextBlob(club_description)
	club_sentiment[club] = club_description_textblob.sentiment.polarity
	club_summary[club] = set(club_description_textblob.words)  # noun_phrases attribute is nice too
	club_summary[club] = club_summary[club] - stoplist  # get rid of common words

print(club_summary)  # dictionary of club name to description 

summary_matrix = []  # making a similarity matrix
for club_a in club_names:  # compare the similarity of club a to all other clubs b
	row = []
	for club_b in club_names:
		# row += [similarities.MatrixSimilarity(club_lsi[club_a][club_lsi[club_b]])]
		row += [(len(club_summary[club_a].intersection(club_summary[club_b])))]  # number of overlapping words
	summary_matrix += [normalize(row)]

clustering = SpectralClustering(n_clusters=20).fit(summary_matrix)

modecluster = max(clustering.labels_, key=list(clustering.labels_).count)

with open('club_cluster_results.txt', 'wt') as outfile:
	writer = csv.writer(outfile, delimiter='\t', lineterminator=os.linesep)
	writer.writerow(['Organization', 'Group_number'])
	for i in range(len(clustering.labels_)):
		writer.writerow([club_names[i], clustering.labels_[i]])
		if clustering.labels_[i] == modecluster:
			print(club_names[i], club_summary[club_names[i]])


