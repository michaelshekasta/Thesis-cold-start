import pickle

import gensim
import nltk
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity

with open('nltk_german_classifier_data.pickle', 'rb') as f:
    tagger = pickle.load(f)


def get_nouns_only(tagger, sentence):
    tagged_sentence = tagger.tag(sentence)
    nouns = []
    for tagged_word in tagged_sentence:
        if tagged_word[1] == u'NN' or tagged_word[1] == u'NE':
            nouns.append(tagged_word[0])
    return nouns


import yoochose_catalog

print('reading catalog')
c = yoochose_catalog.Catalog(
    dir_path="catalog")
items = set(c.get_items())
sentences = []

print('prepare for sentence')

item_nouns = dict()

for index, row in c.catalog_df.iterrows():
    item = str(int(row[c.item_id]))
    try:
        # words = c.description_to_word(row[c.description], only_token=True)
        # words = row[c.description].split(" ")
        words = nltk.word_tokenize(text=row[c.description_html], language='german')
        nouns = get_nouns_only(tagger, words)
        item_nouns[item] = nouns
    except:
        print("item %s has problem check it" % (item))

# words = dict()

# import codecs
#
# fw = codecs.open("item_nouns.csv", "w", "utf-8")
# fw.write(u"item,nouns\n")
# for key, value in item_nouns.iteritems():
#     fw.write(u"%s,%s\n" %(key,' '.join(value)))
#     for word in value:
#         if word in words:
#             words[word] += 1
#         else:
#             words[word] = 1
# fw.close()
#
# fw = codecs.open("dist_nouns.csv", "w", "utf-8")
# fw.write(u"word,count\n")
# for key, value in words.iteritems():
#     fw.write("%s,%s\n" % (key,str(value)))
# fw.close()
# print(item_nouns)

# print(sorted(words.items(), key=operator.itemgetter(1)))

print('training lda')

texts = item_nouns.values()
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000,
                                      passes=1)

print('calc similarity')


def get_vector(vec):
    ans = []
    for i in vec:
        ans.append(i[1])
    return ans


items_key = item_nouns.keys()
with open("sim_lda.csv", "w") as fw:
    fw.write("item1,item2,similarity\n")
    for i in range(len(texts)):
        vec_i = get_vector(lda[dictionary.doc2bow(texts[i])])
        for j in range(i + 1, len(texts)):
            vec_j = get_vector(lda[dictionary.doc2bow(texts[j])])
            try:
                sim = cosine_similarity([vec_i], [vec_j])[0][0]
                if sim > 0.89:
                    fw.write("%s,%s,%s\n" % (items_key[i], items_key[j], str(sim)))
            except:
                print("i=%s, j=%s" % (str(i), str(j)))
