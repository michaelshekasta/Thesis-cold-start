#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io

import config_model
import yoochose_catalog

if __name__ == "__main__":
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    with io.open("catalog_length.csv", 'w', encoding='utf8') as fw:
        print("creating catalog")
        c = yoochose_catalog.Catalog(
            dir_path="catalog", use_german_token=config_model.use_german_tokenizer, mode=1)
        print("write to file")
        for index, row in c.catalog_df.iterrows():
            itemid = str(row.product_id)
            description = c.description_to_word(description=row.description, use_german_token=True, remove_marks=True)
            category = row.Kategorie1.encode('utf-8').strip()
            fw.write(u"%s,%s,%s\n" % (str(itemid), str(category), str(len(description))))
