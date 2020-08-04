#!/usr/bin/env python
# coding: utf-8

# # Surface realization

# In[ ]:


from surface import grammar
from surface import converter
from surface import utils
from collections import defaultdict
import ast
import re
import subprocess
from statistics import stdev, mean

# First we initialize the training and the test file to a variable, the files can be downloaded from the SRST 19 page.

# In[ ]:

trains = ["/home/adaamko/data/UD-train/pt_bosque-ud-train.conllu", "/home/adaamko/data/UD-train/en_ewt-ud-train.conllu", "/home/adaamko/data/UD-train/fr_gsd-ud-train.conllu"]
tests = ["/mnt/store/adaamko/srst2019/T1-test/pt_bosque-ud-test.conllu", "/mnt/store/adaamko/srst2019/T1-test/en_ewt-ud-test.conllu", "/mnt/store/adaamko/srst2019/T1-test/fr_gsd-ud-test.conllu"]


avg_per_lang = []
stop_per_lang = []
std_per_lang = []
sens_len = []

for train, test in zip(trains, tests):
    TRAIN_FILE = train
    TEST_FILE = test

    # Then, we train the two static grammars (the first corresponds to the subgraphs from the ud trees, the second is the fallback grammar, where each rule is binary)
    # 
    # Later, the dynamic grammars are generated from these ones.

    # In[ ]:

    print("Processing: " + test)
    
    grammar.train_subgraphs(TRAIN_FILE, TEST_FILE)
    grammar.train_edges(TRAIN_FILE, TEST_FILE)


    # In[ ]:


    SUBGRAPH_GRAMMAR_FILE = "train_subgraphs"
    EDGE_GRAMMAR_FILE = "train_edges"


    # We need to extract the graphs from the conll format (conversion from conll to isi), and the rules that use the <strong>lin</strong> feature.
    # 
    # The rules are for incorporating the <strong>lin</strong> feature, so we can dynamically delete every rule the contradicts the linearity.

    # In[ ]:


    rules, _ = converter.extract_rules(TEST_FILE)
    graphs, _, id_graphs= converter.convert(TEST_FILE)
    _, sentences, _ = converter.convert(TEST_FILE)
    conll = grammar.get_conll_from_file(TEST_FILE)
    id_to_parse = {}
    stops = []


    # We run through the sentences and call the <strong>alto</strong> parser to generate the derivation and map the ud representation to string.
    # 
    # The alto can be downloaded from [bitbucket](https://bitbucket.org/tclup/alto/downloads/).

    # In[ ]:

    times = []
    lens = []
    for sen_id in range(0, len(rules)):
        print(str(sen_id) + "/" + str(len(rules)))
        try:
            grammar_fn = open('dep_grammar_spec.irtg', 'w') 
            grammar.generate_grammar(SUBGRAPH_GRAMMAR_FILE, rules[sen_id], grammar_fn)
            grammar.generate_terminal_ids(conll[sen_id], grammar_fn)
            grammar_fn.close()
            utils.set_parse("ewt_ones", id_graphs[sen_id])
            out = get_ipython().getoutput('timeout 59 java -Xmx32G -cp alto-2.3.6-SNAPSHOT-all.jar de.up.ling.irtg.script.ParsingEvaluator ewt_ones -g dep_grammar_spec.irtg -I ud -O string=toString -o surface_eval_ewt')
            p_time_s = re.findall("Done, total time: (\d+\.\d+)s", out[-1])
            p_time_ms = re.findall("Done, total time: (\d+) ms", out[-1])
            if p_time_s:
                times.append(float(p_time_s[0]))
            elif p_time_ms:
                times.append(float("0." + p_time_ms[0]))
            text_parse, conll_parse = utils.get_parse("surface_eval_ewt", conll[sen_id])
            lens.append(len(text_parse))
            id_to_parse[sen_id] = (text_parse, conll_parse)
        except StopIteration:
            print("stop iteratioin")
            stops.append(sen_id)
            continue
    
    avg_per_lang.append(mean(times))
    std_per_lang.append(stdev(times))
    stop_per_lang.append(str(len(stops)) + "/" + str(len(rules)))

print("average: " + str(avg_per_lang))
print("std: " + str(std_per_lang))
print("stop: " + str(stop_per_lang))

