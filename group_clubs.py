# Cal Hacks 6.0
# obtain URLs for all organizations
from googlesearch import search
urls = search('site:callink.berkeley.edu inurl:organization/', stop=20)

# open URLs and read in descriptions
club_info = {}
for url in urls:
	page = read(urllib.request(url))
	print(page)
	print(url)
	sys.exit()




# 

from textblob import TextBlob
from sklearn.cluster import SpectralClustering
from gensim import corpora, models, similarities
from collections import defaultdict

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
	club_summary[club] = set(club_description_textblob.noun_phrases)

summary_matrix = []
for club_a in club_names:
	row = []
	for club_b in club_names:
		# row += [similarities.MatrixSimilarity(club_lsi[club_a][club_lsi[club_b]])]
		row += [(len(club_summary[club_a].intersection(club_summary[club_b])))]
	summary_matrix += [normalize(row)]

print(summary_matrix)
clustering = SpectralClustering(n_clusters=2).fit(summary_matrix)
# f = clustering.fit_predict(summary_matrix)
f = clustering.get_params()
print(clustering.labels_)
print(clustering.affinity_matrix_)
print(f)
club_info = {'Student Environmental Resource Center':'The Student Environmental Resource Center (SERC) cultivates a collaborative space to strengthen the collective effectiveness of the sustainability community, and provides resources for students to actualize their visions of a more equitable, socially just, and resilient future. Learn more at serc.berkeley.edu.',
'The Green Initiative Fund':'''Do you...
Need funding for your campus sustainability project?
Want to know what projects have been funded by TGIF?
Want to serve on a committee that awards sustainability grants?
Need a student internship that will provide you with green job skills?
Want to learn more about UC Berkeleys campus sustainability efforts?
Need information regarding starting a green fund at your institution?
If yes, then The Green Initiative Fund is for you!
The Green Initiative Fund (TGIF) provides funding for projects that reduce UC Berkeleys negative impact on the environment and make UC Berkeley more sustainable. TGIF allocates funds to projects that promote sustainable modes of transportation, increase energy and water efficiency, restore habitat, promote environmental and food justice, and reduce the amount of waste created by UC Berkeley. Portions of the fund also support education & behavior change initiatives, student aid (via return to aid), and student internships. Students, faculty, and staff may submit project proposals, which will be selected for funding by an annually appointed grant-making committee (TGIF Committee), consisting of students, faculty, and staff, on which the students have the majority vote.
Learn more at http://tgif.berkeley.edu. and fat dogs!''',
'3DMC':'''3DMC focuses on teaching and learning 3D printing and modeling skills. We hold workshops, participate in competitions, teach DeCals, and host an annual 24-hour 3D printing designathon.  We also tinker with 3D printers and 3D print cool projects!

If you are interested in learning more about 3D modeling/printing, teaching it to other students, or competing in design competitions, then swing by one of our meetings or check out our 3D Printing and Design decal!

Meetings are every Tuesday 7-8 PM in the Makerspace (1st floor Moffitt Library) and fat dogs.'''}



