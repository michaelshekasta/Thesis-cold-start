from sklearn.metrics.pairwise import cosine_similarity

import yoochose_catalog
from item2vec import Item2vec

c = yoochose_catalog.Catalog(
    dir_path="catalog", use_german_token=True)
items = set(c.get_items())

epochs = 25
item2vec = Item2vec(catalog=c, embedding_size=50, hidden_size=10, max_len=10,
                    epoches=epochs)

emb = item2vec.item2emb
sorted_keys = sorted(emb.keys(), reverse=True)
emb_sorted = [emb[key] for key in sorted_keys]

cosine_matrix = cosine_similarity(emb_sorted, emb_sorted)
with open("similarity.csv", "w") as f_write:
    f_write.write("item_i,item_j,score\n")
    for i in range(len(items)):
        for j in range(len(items)):
            if i == j:
                continue
            cosin_ij = cosine_matrix[i][j]
            if cosin_ij > 0.9:
                f_write.write("%s,%s,%s\n" % (sorted_keys[i], sorted_keys[j], cosin_ij))
