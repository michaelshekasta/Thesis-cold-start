from item2vec import Item2vec
import yoochose_catalog
import numpy as np
import random

np.random.seed(0)
random.seed(0)
try:
    import tensorflow as tf

    tf.set_random_seed(7)
except:
    pass
try:
    import os

    os.environ['PYTHONHASHSEED'] = '0'
except:
    pass


class TestItem2Vec:
    def setup_method(self, test_method):
        pass

    def testItem2vec(self):
        c = yoochose_catalog.Catalog(
            dir_path="tests/catalog_test", use_german_token=True)
        item2vec = Item2vec(catalog=c, embedding_size=10, hidden_size=5, max_len=10,
                            epoches=10)
        try:
            import tensorflow as tf
            assert item2vec.item2emb[100169457] == [-2.2042140e-02, 2.4229363e-02, -2.2403020e-02, -1.4319926e-02,
                                                    -1.6100459e-02, 5.7381643e-03, -2.1710992e-05, 4.0693749e-03,
                                                    -1.2456505e-03, -3.3746376e-03]
        except:
            assert item2vec.item2emb[100169457].tolist() == [0.028146304190158844, 0.03664626181125641,
                                                             0.003694504499435425, 0.012484846636652946,
                                                             0.03346257284283638, 0.0038885194808244705,
                                                             0.03074040450155735, -0.04112456738948822,
                                                             -0.02873246930539608, -0.026745563372969627]
