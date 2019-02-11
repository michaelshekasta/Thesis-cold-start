import yoochose_catalog


def test_get_items():
    c = yoochose_catalog.Catalog(
        dir_path="tests/catalog_test", use_german_token=True)
    items = set([100169457, 100169460, 100169463, 100169461])
    items_from_catalog = c.get_items()
    assert len(items.intersection(items_from_catalog)) == 4
    print('test 1')


def test_num_words():
    c = yoochose_catalog.Catalog(
        dir_path="tests/catalog_test", use_german_token=True)
    assert c.n_words == 100
    print('test 2')


def test_cat():
    c = yoochose_catalog.Catalog(
        dir_path="tests/catalog_test", use_german_token=True)
    assert c.n_cat == 1
    print('test 3')
