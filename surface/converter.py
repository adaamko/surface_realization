import sys
import argparse
import json
import os
import re
from collections import defaultdict

from surface.utils import REPLACE_MAP, sanitize_word

def build_dictionaries(filepaths):
    word_to_id = {}
    id_to_word = {}
    graph_data = {}

    word_count = 1
    for filepath in filepaths:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if line == "\n":
                    words = []
                    for w in graph_data:
                        word = graph_data[w]["word"].lower()
                        if word not in word_to_id:
                            word_unique = "WORD" + str(word_count)
                            word_to_id[word] = word_unique
                            id_to_word[word_unique] = word
                            word_count += 1

                    graph_data = {}
                    continue
                if line.startswith("#"):
                    continue
                if line != "\n":
                    fields = line.split("\t")
                    word_id = fields[0]
                    word = fields[2]
                    lemma = fields[1]
                    tree_pos = fields[3]
                    ud_pos = fields[4]
                    mor = fields[5]
                    head = fields[6]
                    ud_edge = fields[7]
                    comp_edge = fields[8]
                    space_after = fields[9]

                    make_default_structure(graph_data, word_id)
                    graph_data[word_id]["word"] = lemma
                    graph_data[word_id]["tree_pos"] = sanitize_word(tree_pos)
                    graph_data[word_id]["mor"] = mor

                    make_default_structure(graph_data, head)
                    graph_data[head]["deps"][word_id] = ud_edge

    return word_to_id, id_to_word




def get_args():
    parser = argparse.ArgumentParser(
        description="Convert conllu file to isi file")
    parser.add_argument("conll_file", type=str, help="path to the CoNLL file")
    return parser.parse_args()



def to_tokenized_output(result_dir, output_dir):
    for filename in os.listdir(result_dir):
        result_filename = os.path.join(result_dir, filename)
        output_filename = os.path.join(
            output_dir, filename.split(".")[0] + ".txt")
        sentences = []
        current_sentence = []
        with open(result_filename, "r") as f:
            for i, line in enumerate(f):
                if line == "\n":
                    sen = " ".join(current_sentence)
                    current_sentence = []
                    sentences.append(sen)
                if line.startswith("#"):
                    continue
                if line != "\n":
                    fields = line.split("\t")
                    word_id = fields[0]
                    word = fields[2]
                    current_sentence.append(word)

        with open(output_filename, "w") as f:
            for i, sentence in enumerate(sentences):
                # if i > 942:
                    # f.write("#sent_id = " + str(i-943+1) + "\n")
                    # f.write("# text = " + sentence + "\n")
                    # f.write("\n")
                f.write("#sent_id = " + str(i+1) + "\n")
                f.write("#text = " + sentence + "\n")
                f.write("\n")



def extract_rules(dev, word_to_id):
    graph_data = {}
    noun_list = []
    id_to_rules = defaultdict(list)
    id_to_sentence = {}
    sentences = 0
    with open(dev, "r") as f:
        for i, line in enumerate(f):
            if line == "\n":
                words = []
                for w in graph_data:
                    words.append(graph_data[w]["word"])
                    subgraphs = {"root": None, "graph": []}
                    rules = []
                    if "tree_pos" not in graph_data[w]:
                        subgraphs["root"] = "ROOT"
                        for dep in graph_data[w]["deps"]:
                            to_pos = graph_data[dep]["tree_pos"]
                            word = graph_data[dep]["word"].lower()
                            subgraphs["graph"].append(
                                {"to": (word.lower(), to_pos), "edge": "root", "dir": None})
                        id_to_rules[sentences].append(subgraphs)
                        continue

                    subgraphs["root"] = (
                        graph_data[w]["word"].lower(), graph_data[w]["tree_pos"])

                    for dep in graph_data[w]["deps"]:
                        edge_dep = graph_data[w]["deps"][dep]
                        to_pos = graph_data[dep]["tree_pos"]
                        word = graph_data[dep]["word"].lower()
                        mor = graph_data[dep]["mor"]

                        if "tree_pos" in graph_data[w]:
                            if "lin=+" in mor:
                                subgraphs["graph"].append(
                                    {"to": (word.lower(), to_pos), "edge": edge_dep.replace(":", "_"), "dir": "S"})
                            elif "lin=-" in mor:
                                subgraphs["graph"].append(
                                    {"to": (word.lower(), to_pos), "edge": edge_dep.replace(":", "_"), "dir": "B"})
                            else:
                                subgraphs["graph"].append(
                                    {"to": (word.lower(), to_pos), "edge": edge_dep.replace(":", "_"), "dir": None})

                    id_to_rules[sentences].append(subgraphs)
                graph_data = {}
                noun_list = []
                sentences += 1
                continue
            if line.startswith("# text"):
                id_to_sentence[sentences] = line.strip()
            if line.startswith("#"):
                continue
            if line != "\n":
                fields = line.split("\t")
                word_id = fields[0]
                word = fields[2]
                lemma = fields[1]
                tree_pos = fields[3]
                ud_pos = fields[4]
                mor = fields[5]
                head = fields[6]
                ud_edge = fields[7]
                comp_edge = fields[8]
                space_after = fields[9]

                make_default_structure(graph_data, word_id)
                graph_data[word_id]["word"] = word_to_id[lemma.lower()]
                graph_data[word_id]["tree_pos"] = sanitize_word(tree_pos)
                graph_data[word_id]["mor"] = mor

                make_default_structure(graph_data, head)
                graph_data[head]["deps"][word_id] = ud_edge
    return id_to_rules, id_to_sentence


def print_output(graph_data, graph_root):
    print(make_graph_string(graph_data, graph_root))


def make_id_graph(graph_data, word_id, word_to_id):
    graph_string = "({1}_{0} / {1}_{0}".format(str(word_id),
                                               word_to_id[graph_data[word_id]["word"]])
    for other_id in graph_data[word_id]["deps"]:
        edge = graph_data[word_id]["deps"][other_id]
        graph_string += ' :{0} '.format(edge.replace(':', '_'))
        graph_string += make_id_graph(graph_data, other_id, word_to_id)
    graph_string += ")"
    return graph_string


def make_graph_string(graph_data, word_id):
    graph_string = "({0} / {0}".format(graph_data[word_id]["word"])
    for other_id in graph_data[word_id]["deps"]:
        edge = graph_data[word_id]["deps"][other_id]
        graph_string += ' :{0} '.format(edge.replace(':', '_'))
        graph_string += make_graph_string(graph_data, other_id)
    graph_string += ")"
    return graph_string


def sanitize_pos(pos):
    if pos == "HYPH":
        pos = "PUNCT"

    pos = pos.replace("|", "PIPE")
    pos = pos.replace("=", "EQUAL")

    is_punct = True
    for character in pos:
        if character not in REPLACE_MAP:
            is_punct = False

    if is_punct == True:
        return "PUNCT"
    else:
        return pos


def convert(conll_file, word_to_id):
    sentences = []
    graphs = []
    words = defaultdict(int)
    id_to_sentences = {}
    id_to_graph = {}
    id_to_idgraph = {}
    with open(conll_file) as conll_file:
        graph_data = {}
        graph_root = "0"
        sen_id = 0
        for line in conll_file:
            if line == "\n":
                print(json.dumps(graph_data))
                graph = make_graph_string(graph_data, graph_root)
                id_graph = make_id_graph(graph_data, graph_root, word_to_id)
                graphs.append(graph)
                id_to_graph[sen_id] = graph
                id_to_idgraph[sen_id] = id_graph
                graph_data = {}
                graph_root = "0"
                words = defaultdict(int)
                sen_id += 1
                continue
            if line.startswith("# text ="):
                sentence = line.split("=")[1]
                graphs.append(line.strip())
                id_to_sentences[sen_id] = line.strip()
                continue
            elif line.startswith("#") or not line:
                continue

            fields = line.split("\t")
            dep_word_id = fields[0]

            dep_word = fields[1].lower()
            words[dep_word] += 1

            tree_pos = sanitize_word(sanitize_pos(fields[3]))
            ud_pos = fields[4]
            root_word_id = fields[6]
            ud_edge = fields[7]

            make_default_structure(graph_data, dep_word_id)
            graph_data[dep_word_id]["word"] = dep_word
            graph_data[dep_word_id]["tree_pos"] = tree_pos
            graph_data[dep_word_id]["ud_pos"] = sanitize_word(ud_pos)

            """
            for the head; store the edges with the head of the dependency
            """
            # Ignore :root dependencies,
            # but remember the root word of the graph
            if "0" != root_word_id:
                make_default_structure(graph_data, root_word_id)
                graph_data[root_word_id]["deps"][dep_word_id] = ud_edge
            else:
                graph_root = dep_word_id

    with open("ewt_graphs", "w") as f:
        f.write("# IRTG unannotated corpus file, v1.0\n")
        f.write("# interpretation ud: de.up.ling.irtg.algebra.graph.GraphAlgebra\n")
        for graph in graphs:
            f.write(graph + "\n")
    with open("ewt_sentences", "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    return id_to_graph, id_to_sentences, id_to_idgraph


def make_default_structure(graph_data, word_id):
    if word_id not in graph_data:
        graph_data[word_id] = {
            "word": "",
            "deps": {},
        }


def main():
    args = get_args()
    convert(args.conll_file)


if __name__ == "__main__":
    main()
