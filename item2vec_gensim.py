from gensim import models

import yoochose_catalog

print('reading catalog')
c = catalog.Catalog(
    dir_path="catalog", use_german_token=True)
# c = catalog.Catalog(
#     dir_path="catalog")
items = set(c.get_items())
sentences = []

print('prepare for sentence')

for index, row in c.catalog_df.iterrows():
    item = str(row[c.item_id])
    words = c.description_to_word(row[c.description])
    sentence = models.doc2vec.TaggedDocument(words=words, tags=[str(item)])
    sentences.append(sentence)

print('prepare for training')

model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1, seed=0, size=50, window=5, iter=20)
model.build_vocab(sentences)

print('training')
model.train(sentences, total_examples=len(sentences), epochs=50, compute_loss=True)

print('save models')
model.save("my_model.doc2vec")

with open("doc2vec_sim.csv", "w") as fw:
    fw.write("item1,item2,sim\n")
    for item in items:
        sim_items = model.docvecs.most_similar(str(item))
        for sim_item in sim_items:
            fw.write("%s,%s,%s\n" % (str(item), str(sim_item[0]), str(sim_item[1])))

# print models.docvecs["100004774"]
# print models.docvecs.most_similar(["100004774"])
# print models.docvecs.most_similar(["100001180"])
# print models.docvecs.most_similar(["100007658"])
# print models.docvecs.most_similar(["100000505"])
# print models.docvecs.most_similar(["100000507"])
# print models.docvecs.most_similar(["100000067"])
# print models.docvecs.most_similar(["100001075"])
# print models.docvecs.most_similar(["100004442"])
# print models.docvecs.most_similar(["100004770"])
# print models.docvecs.most_similar(["100007694"])
# print models.docvecs.most_similar(["100005334"])


# model_loaded = models.Doc2Vec.load('my_model.doc2vec')
# print model_loaded.docvecs["100004774"]
# # print model_loaded.docvecs.most_similar(["100004774"])
# # print model_loaded.docvecs.most_similar(["100001180"])
# # print model_loaded.docvecs.most_similar(["100007658"])
# # print model_loaded.docvecs.most_similar(["100000505"])
# # print model_loaded.docvecs.most_similar(["100000507"])
# print model_loaded.docvecs.most_similar(["100000067"])
# print model_loaded.docvecs.most_similar(["100001075"])
# print model_loaded.docvecs.most_similar(["100004442"])
# print model_loaded.docvecs.most_similar(["100004770"])
# print model_loaded.docvecs.most_similar(["100007694"])
# print model_loaded.docvecs.most_similar(["100005334"])
# print model_loaded.docvecs.most_similar(["SENT_1"])
# print model_loaded.docvecs.most_similar(["SENT_2"])
