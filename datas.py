import pandas as pd
import Config as cg
import json
from random import shuffle
from collections import Counter


def load_data():

    data = json.load(open(cg.data_path))

    return data

def res_data():
    data = load_data()

    mention_domain = [r['domain'] for r in data]
    counter = Counter(mention_domain)
    sorted_items_by_value = sorted(counter.items(), key=lambda item: item[1], reverse=True)  

    # 按照domains的数量排序 取后三
    low_resource_domains = []
    for i in range(0,3):
        low_resource_domains.append(sorted_items_by_value[-3:][i][0])

    train_examples = [r for r in data if r['domain'] not in low_resource_domains]
    test_examples = [r for r in data if r['domain'] in low_resource_domains]

    return train_examples, test_examples
