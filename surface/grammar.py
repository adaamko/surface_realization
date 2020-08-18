import sys
import re
import copy
import operator
import argparse
import converter
from itertools import chain, combinations, product
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from utils import all_subsets

ENGLISH_WORD = re.compile("^[a-zA-Z0-9]*$")

REPLACE_MAP = {
    ":": "COLON",
    ",": "COMMA",
    ".": "PERIOD",
    ";": "SEMICOLON",
    "-": "HYPHEN",
    "_": "DASH",
    "[": "LSB",
    "]": "RSB",
    "(": "LRB",
    ")": "RRB",
    "{": "LCB",
    "}": "RCB",
    "!": "EXC",
    "?": "QUE",
    "'": "SQ",
    '"': "DQ",
    "/": "PER",
    "\\": "BSL",
    "#": "HASHTAG",
    "%": "PERCENT",
    "&": "ET",
    "@": "AT",
    "$": "DOLLAR",
    "*": "ASTERISK",
    "^": "CAP",
    "`": "IQ",
    "+": "PLUS",
    "|": "PIPE",
    "~": "TILDE",
    "<": "LESS",
    ">": "MORE",
    "=": "EQ"
}

KEYWORDS = set(["feature"])


class Grammar():

    def __init__(self):
        super().__init__()
        self.subgraphs_highest = {}
        self.subgraphs = OrderedDict()

    def make_default_structure(self, graph_data, word_id):
        if word_id not in graph_data:
            graph_data[word_id] = {
                "word": "",
                "deps": {},
            }

    def add_unseen_rules(self, grammar, fn_dev):
        graph_data = {}
        noun_list = []
        with open(fn_dev, "r") as f:
            for i, line in enumerate(f):
                if line == "\n":
                    for w in graph_data:
                        for dep in graph_data[w]["deps"]:
                            edge_dep = graph_data[w]["deps"][dep]
                            to_pos = graph_data[dep]["tree_pos"]
                            mor = graph_data[dep]["mor"]

                            if "tree_pos" in graph_data[w]:
                                line_key_before = graph_data[w]["tree_pos"] + \
                                    ">" + to_pos + "|" + edge_dep + "&>"
                                line_key_after = graph_data[w]["tree_pos"] + \
                                    ">>" + to_pos + "|" + edge_dep + "&"

                                if "lin=+" in mor and line_key_after in grammar:
                                    grammar[line_key_after] += 1
                                elif "lin=-" in mor and line_key_before in grammar:
                                    grammar[line_key_before] += 1

                                if line_key_before not in grammar and line_key_after not in grammar:
                                    if "lin=+" in mor:
                                        grammar[line_key_after] = 1
                                    elif "lin=-" in mor:
                                        grammar[line_key_before] = 1
                                    else:
                                        grammar[line_key_before] = 1
                                        grammar[line_key_after] = 1
                                elif line_key_before not in grammar:
                                    grammar[line_key_before] = 1
                                elif line_key_after not in grammar:
                                    grammar[line_key_after] = 1

                    graph_data = {}
                    noun_list = []
                    continue
                if line != "\n":
                    fields = line.split("\t")
                    word_id = fields[0]
                    word = fields[1]
                    tree_pos = fields[3]
                    ud_pos = fields[4]
                    mor = fields[5]
                    head = fields[6]
                    ud_edge = fields[7]

                    self.make_default_structure(graph_data, word_id)
                    graph_data[word_id]["word"] = word
                    graph_data[word_id]["tree_pos"] = self.sanitize_word(
                        ud_pos)
                    graph_data[word_id]["mor"] = mor

                    self.make_default_structure(graph_data, head)
                    graph_data[head]["deps"][word_id] = ud_edge

    def train_subgraphs(self, fn_train, fn_dev):
        graph_data = {}
        noun_list = []
        pos_to_order = defaultdict(set)
        order_to_count = defaultdict(lambda: 1)

        with open(fn_train, "r") as f:
            for i, line in tqdm(enumerate(f)):
                if line.startswith("#"):
                    continue
                if line == "\n":
                    for w in graph_data:
                        self.count_on_graph(
                            graph_data, w, pos_to_order, order_to_count)

                    graph_data = {}
                    noun_list = []
                    continue
                if line != "\n":
                    fields = line.split("\t")
                    word_id = fields[0]
                    word = fields[1]
                    lemma = fields[2]
                    tree_pos = fields[3]
                    ud_pos = fields[4]
                    mor = fields[5]
                    head = fields[6]
                    ud_edge = fields[7]

                    self.make_default_structure(graph_data, word_id)
                    graph_data[word_id]["word"] = word
                    graph_data[word_id]["lemma"] = lemma
                    graph_data[word_id]["tree_pos"] = self.sanitize_word(
                        tree_pos)
                    graph_data[word_id]["mor"] = mor

                    self.make_default_structure(graph_data, head)
                    graph_data[head]["deps"][word_id] = ud_edge

        sorted_x = sorted(order_to_count.items(),
                          key=operator.itemgetter(1), reverse=True)
        sorted_dict = OrderedDict(sorted_x)
        # self.add_unseen_rules(sorted_dict, fn_dev)
        self.subgraphs = sorted_dict

        for order_key in pos_to_order:
            order_set = pos_to_order[order_key]
            max_key = max(order_set, key=lambda x: order_to_count[x])
            pos_to_order[order_key] = max_key

        self.subgraphs_highest = pos_to_order

    def count_on_graph(self, graph_data, w, pos_to_order, order_to_count):
        deps = graph_data[w]["deps"]
        possibility = 0
        for subset in all_subsets(list(deps.keys())):
            if len(subset) > 5:
                return
            nodes_before = []
            nodes_after = []
            for dep in subset:
                if "tree_pos" in graph_data[w]:
                    if int(dep) < int(w):
                        nodes_before.append(int(dep))
                    elif int(dep) > int(w):
                        nodes_after.append(int(dep))

            s_nodes_before = sorted(nodes_before)
            s_nodes_after = sorted(nodes_after)
            nodes = sorted(s_nodes_before + s_nodes_after)

            # From the IDS, get (lemma, pos) pairs, so after the descartes product of the list can be calculated
            nodes_paired = [[graph_data[str(node)]["lemma"].lower(
            ), graph_data[str(node)]["tree_pos"]] for node in nodes]

            for combined in product(*nodes_paired):
                possibility += 1
                for i, elem in enumerate(combined):
                    graph_data[str(nodes[i])]["element"] = elem

                pos_line_key = ""
                pos_order_key = ""

                lemma_line_key = ""
                lemma_order_key = ""

                if "tree_pos" not in graph_data[w]:
                    pos_line_key += ">"
                    lemma_line_key += ">"
                else:
                    pos_line_key += graph_data[w]["tree_pos"] + ">"
                    pos_order_key += graph_data[w]["tree_pos"] + ">"

                    lemma_line_key += graph_data[w]["lemma"].lower() + ">"
                    lemma_order_key += graph_data[w]["lemma"].lower() + ">"

                for n in s_nodes_before:
                    n = str(n)
                    edge = graph_data[w]["deps"][n]
                    pos = graph_data[n]["element"]
                    lemma = graph_data[n]["element"]

                    pos_line_key += pos + "|" + edge + "&"
                    lemma_line_key += lemma + "|" + edge + "&"
                pos_line_key += ">"
                lemma_line_key += ">"

                for n in s_nodes_after:
                    n = str(n)
                    edge = graph_data[w]["deps"][n]
                    pos = graph_data[n]["element"]
                    lemma = graph_data[n]["element"]

                    pos_line_key += pos + "|" + edge + "&"
                    lemma_line_key += lemma + "|" + edge + "&"

                nodes.sort(key=lambda x: graph_data[str(x)]["element"])
                for n in nodes:
                    n = str(n)
                    edge = graph_data[w]["deps"][n]
                    pos = graph_data[n]["element"]
                    lemma = graph_data[n]["element"]

                    pos_order_key += pos + "|" + edge + "&"
                    lemma_order_key += lemma + "|" + edge + "&"

                pos_to_order[pos_order_key.strip("&")].add(
                    pos_line_key.strip("&"))
                pos_to_order[lemma_order_key.strip("&")].add(
                    lemma_line_key.strip("&"))
                order_to_count[pos_line_key.strip("&")] += 1
                order_to_count[lemma_line_key.strip("&")] += 1

    def sanitize_word(self, word):
        for pattern, target in REPLACE_MAP.items():
            word = word.replace(pattern, target)
        for digit in "0123456789":
            word = word.replace(digit, "DIGIT")
        if word in KEYWORDS:
            word = word.upper()
        return word

    def generate_terminal_ids(self, conll, grammar_fn):
        TEMPLATE = (
            '{0} -> {0}_{1}\n[string] {0}_{1}\n[ud] "({0}_{1}<root> / {0}_{1})"\n')

        for w_id in conll:
            print(TEMPLATE.format(self.sanitize_word(
                conll[w_id][3]), w_id), file=grammar_fn)

    def generate_terminals(self, fn, grammar_fn):
        TEMPLATE = (
            '{0} -> {1}_{2}_{0}\n[string] {1}\n[ud] "({1}_{2}<root> / {1}_{2})"\n')

        with open(fn) as train_file:
            terminals = set()
            words = defaultdict(int)
            for line in train_file:
                if line.startswith("#"):
                    continue
                if line.strip():
                    fields = line.split("\t")
                    word = self.sanitize_word(fields[1])
                    if ENGLISH_WORD.match(word):
                        terminals.add(
                            word + "_" + str(words[word]) + "_" + fields[3])
                        words[word] += 1
                elif not line.strip():
                    words = defaultdict(int)

        for terminal in terminals:
            t_field = terminal.split("_")
            print(TEMPLATE.format(t_field[2],
                                  t_field[0], t_field[1]), file=grammar_fn)

    def generate_grammar(self, rules, grammar_fn, binary=False):
        start_rule_set = set()
        print("interpretation string: de.up.ling.irtg.algebra.StringAlgebra",
              file=grammar_fn)
        print(
            "interpretation ud: de.up.ling.irtg.algebra.graph.GraphAlgebra",
            file=grammar_fn)
        print("\n", file=grammar_fn)

        trained_edges = self.subgraphs

        # frequencies = [int(trained_edges[subgraph])
        #                for subgraph in trained_edges]
        # freq_sums = sum(frequencies)

        self.query_rules(rules, grammar_fn)

    def query_rules(self, rules, grammar_fn):
        counter = 1
        for graph in rules:
            if graph["root"] != "ROOT":
                subgraph_nodes = []
                # subgraph_nodes.append(graph["root"])
                subgraph_edges = []
                subgraph_rules = []

                for e in graph["graph"]:
                    subgraph_nodes.append([e["to"], e["edge"]])

                for subset in all_subsets(subgraph_nodes):
                    if len(subset) > 5:
                        return
                    nodes = [node[0] for node in subset]
                    edges = [node[1] for node in subset]
                    for combined in product(*list(nodes)):
                        sorted_nodes_edges = [(x, y) for x, y in sorted(
                            zip(list(combined), edges), key=lambda pair: pair[0])]
                        query_string = "&".join(
                            ["|".join(x) for x in sorted_nodes_edges])
                        pos_query_string = graph["root"][1] + \
                            ">" + query_string
                        lemma_query_string = graph["root"][0] + \
                            ">" + query_string

                        if pos_query_string in self.subgraphs_highest:
                            subgraph = self.subgraphs_highest[pos_query_string]
                            fields = subgraph.split(">")
                            head = fields[0]
                            dep_before = fields[1].replace(":", "_").strip("&")
                            dep_after = fields[2].replace(":", "_").strip("&")
                            self.print_rules(
                                head,
                                dep_before,
                                dep_after,
                                counter,
                                grammar_fn)
                            counter += 1

                        if lemma_query_string in self.subgraphs_highest:
                            subgraph = self.subgraphs_highest[lemma_query_string]
                            fields = subgraph.split(">")
                            head = fields[0]
                            dep_before = fields[1].replace(":", "_").strip("&")
                            dep_after = fields[2].replace(":", "_").strip("&")
                            self.print_rules(
                                head,
                                dep_before,
                                dep_after,
                                counter,
                                grammar_fn)
                            counter += 1
            else:
                start_rule_set = set()
                for e in graph["graph"]:
                    for element in e["to"]:
                        start_rule_set.add(element)
                self.print_start_rule(start_rule_set, grammar_fn)

    def remove_bidirection(self, id_to_rules):
        graphs_with_dirs = {}
        id_to_direction = {}

        for ind in id_to_rules:
            for i, graph in enumerate(id_to_rules[ind]):
                dict_key = tuple(sorted(graph.items()))[1:]
                if dict_key not in graphs_with_dirs:
                    graphs_with_dirs[dict_key] = (ind, i)
                    id_to_direction[(ind, i)] = graph["dir"]
                else:
                    graph_id = graphs_with_dirs[dict_key]
                    if id_to_direction[graph_id] != graph["dir"]:
                        id_to_rules[ind][i]["dir"] = None
                        id_to_rules[graph_id[0]][graph_id[1]]["dir"] = None

    def print_rules(self,
                    h,
                    d_before,
                    d_after,
                    counter,
                    grammar_fn):
        rewrite_rule = h + " -> rule_" + str(counter) + "(" + h + ","
        if not d_before and not d_after:
            return

        before_nodes = []
        before_edges = []
        after_nodes = []
        after_edges = []

        if d_before:
            for n in d_before.split("&"):
                n = n.split("|")
                rewrite_rule += n[0] + ","
                before_nodes.append(n[0])
                before_edges.append(n[1])

        if d_after:
            for n in d_after.split("&"):
                n = n.split("|")
                rewrite_rule += n[0] + ","
                after_nodes.append(n[0])
                after_edges.append(n[1])

        rewrite_rule = rewrite_rule.strip(",")
        rewrite_rule += ") "
        rewrite_rule += "[0.99]"

        print(rewrite_rule, file=grammar_fn)
        self.generate_string_line(h, before_nodes, after_nodes, grammar_fn)
        self.generate_graph_line(before_edges, after_edges, grammar_fn)
        print(file=grammar_fn)

    def generate_string_line(self, h, before_nodes, after_nodes, grammar_fn):
        string_temp = '[string] *({0})'
        nodes = []
        for i, node in enumerate(before_nodes):
            nodes.append("?" + str(i + 2))

        nodes.append("?1")

        for i, node in enumerate(after_nodes):
            nodes.append("?" + str(i + len(before_nodes) + 2))

        pairs = copy.deepcopy(nodes)

        while len(pairs) != 2:
            copy_pairs = []
            if len(pairs) % 2 == 0:
                for n in range(1, len(pairs), 2):
                    copy_pairs.append(
                        "*(" + str(pairs[n - 1]) + "," + str(pairs[n]) + ")")
            elif len(pairs) % 2 == 1:
                for n in range(1, len(pairs), 2):
                    copy_pairs.append(
                        "*(" + str(pairs[n - 1]) + "," + str(pairs[n]) + ")")
                copy_pairs.append(pairs[-1])

            pairs = copy_pairs

        string_line = string_temp.format(pairs[0] + "," + pairs[1])

        print(string_line, file=grammar_fn)

    def generate_graph_line(self, before_edges, after_edges, grammar_fn):
        graph_line = "[ud] "
        edges = before_edges + after_edges

        if not edges:
            return
        for i in reversed(range(len(edges))):
            graph_line += "f_dep" + str(i + 1) + "("
        for i in range(len(edges)):
            graph_line += "merge("
        graph_line += "merge("
        graph_line += '?1,"(r<root> '

        for i, edge in enumerate(edges):
            graph_line += ":" + edge + " " + \
                "(d" + str(i + 1) + "<dep" + str(i + 1) + ">) "
        graph_line = graph_line.strip()
        graph_line += ')"), '

        for i, edge in enumerate(edges):
            graph_line += "r_dep" + str(i + 1) + "(?" + str(i + 2) + ")), "
        graph_line = graph_line.strip().strip(",")

        for i in range(len(edges)):
            graph_line += ")"
        print(graph_line, file=grammar_fn)

    def print_start_rule(self, s, grammar_fn):
        for i in s:
            print("S! -> start_b_{}({}) [1.0]".format(i, i), file=grammar_fn)
            print("[string] ?1", file=grammar_fn)
            print("[ud] ?1", file=grammar_fn)
            print(file=grammar_fn)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and generate IRTG parser")
    parser.add_argument("--train_file", type=str,
                        help="path to the CoNLL train file")
    parser.add_argument("--test_file", type=str,
                        help="path to the CoNLL test file")
    return parser.parse_args()


def main():
    args = get_args()
    grammar = Grammar()
    grammar.train_subgraphs(args.train_file, args.test_file)
    rules, _ = converter.extract_rules(args.test_file)
    graphs, _, id_graphs = converter.convert(args.test_file)
    _, sentences, _ = converter.convert(args.test_file)
    conll = converter.get_conll_from_file(args.test_file)
    id_to_parse = {}
    stops = []
    grammar_fn = open('dep_grammar_spec.irtg', 'w')
    grammar.generate_grammar(rules[0], grammar_fn)


if __name__ == "__main__":
    main()
