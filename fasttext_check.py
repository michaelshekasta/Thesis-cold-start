#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import io

import config_model
import system_utils
import yoochose_catalog


def create_files(mode):
    print('starting mode %d...' % mode)
    print('create catalog...')
    c = yoochose_catalog.Catalog(dir_path="catalog", use_german_token=config_model.use_german_tokenizer, mode=mode, to_replace_umlauts=False, clean_words_punctuation=False)

    words_wrote_in_catalog = set()
    words = c.words
    filename_read = "C:\\Users\\Michael\\Downloads\\wiki.de.vec"
    filename_write = "C:\\Users\\Michael\\Downloads\\michael.de.vec%d" % mode
    num_line = 0
    with codecs.open(filename_read, 'r', encoding='utf8') as f:
        with io.open(filename_write, 'w', encoding='utf8') as fw:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                line_split = line.split(" ")
                if line_split[0] in words:
                    fw.write(line)
                    num_line += 1
                    words_wrote_in_catalog.add(line_split[0])
    print(set(words).difference(words_wrote_in_catalog))
    message = "mode = %d , lines wrote = %d, words_wrote_in_catalog = %d ,words in catalog = %d" % (
    mode, num_line, len(words_wrote_in_catalog), len(set(words)))
    print(message)
    system_utils.send_email(subject='extract vectors',body=message)


# for i in range(1, 8):
#     create_files(i)
create_files(2)
