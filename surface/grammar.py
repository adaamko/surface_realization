import argparse
import copy
import logging
import operator
import pickle
import requests
import sys
from itertools import product
from tqdm import tqdm
from collections import defaultdict, OrderedDict

from flask import Flask
from flask import request
from flask import jsonify

from surface import converter
from surface.utils import all_subsets


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

    @staticmethod
    def load(model_file):
        with open(model_file, 'rb') as f:
            grammar = pickle.load(f)
            return grammar

    def save(self, model_file):
        with open(model_file, 'wb') as f:
            pickle.dump(self, f)

    def make_default_structure(self, graph_data, word_id):
        if word_id not in graph_data:
            graph_data[word_id] = {
                "word": "",
                "deps": {},
            }

    def build_dictionaries(self, files):
        self.word_to_id, self.id_to_word = converter.build_dictionaries(files)

    def train_subgraphs(self, fn_train, max_subset_size):
        graph_data = {}
        # noun_list = []
        pos_to_order = defaultdict(set)
        order_to_count = defaultdict(lambda: 1)

        with open(fn_train, "r") as f:
            for i, line in tqdm(enumerate(f)):
                if line.startswith("#"):
                    continue
                if line == "\n":
                    for w in graph_data:
                        self.count_on_graph(
                            graph_data, w, pos_to_order, order_to_count,
                            max_subset_size)

                    graph_data = {}
                    # noun_list = []
                    continue
                if line != "\n":
                    fields = line.split("\t")
                    word_id = fields[0]
                    word = fields[2]
                    lemma = fields[1]
                    tree_pos = fields[3]
                    # ud_pos = fields[4]
                    mor = fields[5]
                    head = fields[6]
                    ud_edge = fields[7]

                    self.make_default_structure(graph_data, word_id)
                    graph_data[word_id]["word"] = word
                    graph_data[word_id]["lemma"] = self.word_to_id[lemma.lower()]  # noqa
                    graph_data[word_id]["tree_pos"] = self.sanitize_word(
                        tree_pos)
                    graph_data[word_id]["mor"] = int(
                        mor.split("|")[-1].split("original_id=")[1])

                    self.make_default_structure(graph_data, head)
                    graph_data[head]["deps"][word_id] = ud_edge

        sorted_x = sorted(order_to_count.items(),
                          key=operator.itemgetter(1), reverse=True)
        sorted_dict = OrderedDict(sorted_x)
        self.subgraphs = sorted_dict

        for order_key in pos_to_order:
            order_set = pos_to_order[order_key]
            max_key = sorted(
                list(order_set), key=lambda x: order_to_count[x], reverse=True)
            pos_to_order[order_key] = max_key

        self.subgraphs_highest = pos_to_order

    def count_on_graph(
            self, graph_data, w, pos_to_order, order_to_count,
            max_subset_size):
        deps = graph_data[w]["deps"]
        possibility = 0
        for subset in all_subsets(list(deps.keys())):
            if len(subset) > max_subset_size:
                return
            nodes_before = []
            nodes_after = []
            for dep in subset:
                if "tree_pos" in graph_data[w]:
                    if int(graph_data[dep]["mor"]) < int(graph_data[w]["mor"]):
                        nodes_before.append(int(dep))
                    elif int(graph_data[dep]["mor"]) > int(graph_data[w]["mor"]):  # noqa
                        nodes_after.append(int(dep))

            s_nodes_before = sorted(
                nodes_before, key=lambda x: int(graph_data[str(x)]["mor"]))
            s_nodes_after = sorted(
                nodes_after, key=lambda x: int(graph_data[str(x)]["mor"]))
            nodes = sorted(s_nodes_before + s_nodes_after,
                           key=lambda x: int(graph_data[str(x)]["mor"]))

            # From the IDS, get (lemma, pos) pairs, so after the descartes
            # product of the list can be calculated
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
                    edge = graph_data[w]["deps"][n].replace(":", "_")
                    pos = graph_data[n]["element"]
                    lemma = graph_data[n]["element"]

                    pos_line_key += pos + "|" + edge + "&"
                    lemma_line_key += lemma + "|" + edge + "&"
                pos_line_key += ">"
                lemma_line_key += ">"

                for n in s_nodes_after:
                    n = str(n)
                    edge = graph_data[w]["deps"][n].replace(":", "_")
                    pos = graph_data[n]["element"]
                    lemma = graph_data[n]["element"]

                    pos_line_key += pos + "|" + edge + "&"
                    lemma_line_key += lemma + "|" + edge + "&"

                nodes.sort(key=lambda x: graph_data[str(x)]["element"])
                for n in nodes:
                    n = str(n)
                    edge = graph_data[w]["deps"][n].replace(":", "_")
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

    def generate_terminal_ids(self, sen):
        TEMPLATE = "\n".join([
            '{0} -> {0}_{1}',
            '[string] {0}_{1}',
            '[ud] "({0}_{1}<root> / {0}_{1})"',
            ''])

        POS_TEMPLATE = "\n".join([
            '{0} -> pos_to_word_{1}({2}) [0.99]',
            '[string] ?1',
            '[ud] ?1',
            ''])

        rules = []

        for tok in sen['words']:
            word_id = self.word_to_id[tok['lemma'].lower()]
            template = TEMPLATE.format(word_id, tok['id'])
            pos_template = POS_TEMPLATE.format(
                tok['pos'], tok['id'], word_id)
            rules.append(template)
            rules.append(pos_template)

        for rule in rules:
            yield rule

    def gen_header(self):
        yield "interpretation string: de.up.ling.irtg.algebra.StringAlgebra"
        yield "interpretation ud: de.up.ling.irtg.algebra.graph.GraphAlgebra"
        yield "\n"

    def generate_grammar(self, rules, max_subset_size, binary=False):
        yield from self.gen_header()
        yield from self.gen_all_rules(rules, binary, max_subset_size)

    def query_order(self, constraints, key):
        if not constraints:
            return self.subgraphs_highest[key][0]
        else:
            for subgraph in self.subgraphs_highest[key]:
                fields = subgraph.split(">")
                # head = fields[0]
                dep_before = fields[1].replace(":", "_").strip("&")
                dep_after = fields[2].replace(":", "_").strip("&")
                before_nodes = []
                after_nodes = []

                if dep_before:
                    for n in dep_before.split("&"):
                        n = n.split("|")
                        before_nodes.append(n[0])

                if dep_after:
                    for n in dep_after.split("&"):
                        n = n.split("|")
                        after_nodes.append(n[0])
                subgraph_ok = True
                for constrain in constraints:
                    if constrain["dir"] == "S" and (
                            constrain["to"][0] in before_nodes or
                            constrain["to"][1] in before_nodes):
                        subgraph_ok = False
                        break
                    if constrain["dir"] == "B" and (
                            constrain["to"][0] in after_nodes or
                            constrain["to"][1] in after_nodes):
                        subgraph_ok = False
                        break

                if subgraph_ok:
                    return subgraph

            return self.subgraphs_highest[key][0]

    def gen_subgraph_rules(self, subgraph, counter):
        fields = subgraph.split(">")
        head = fields[0]
        dep_before = fields[1].replace(":", "_").strip("&")
        dep_after = fields[2].replace(":", "_").strip("&")
        yield from self.gen_rules(head, dep_before, dep_after, counter)

    def gen_all_rules(self, rules, binary, max_subset_size):
        counter = 1
        for graph in rules:
            if graph["root"] != "ROOT":
                subgraph_nodes = []
                # subgraph_nodes.append(graph["root"])
                constraints = []

                for e in graph["graph"]:
                    subgraph_nodes.append([e["to"], e["edge"]])
                    if e["dir"]:
                        constraints.append(e)
                for subset in all_subsets(subgraph_nodes):
                    if binary and len(subset) > 1:
                        break
                    if not binary and len(subset) > max_subset_size:
                        break
                    nodes = [node[0] for node in subset]
                    edges = [node[1] for node in subset]
                    for combined in product(*list(nodes)):
                        sorted_nodes_edges = [(x, y) for x, y in sorted(
                            zip(list(combined), edges),
                            key=lambda pair: pair[0])]
                        query_string = "&".join(
                            ["|".join(x) for x in sorted_nodes_edges])
                        pos_query_string = graph["root"][1] + \
                            ">" + query_string
                        lemma_query_string = graph["root"][0] + \
                            ">" + query_string

                        if pos_query_string in self.subgraphs_highest:
                            subgraph = self.query_order(
                                constraints, pos_query_string)
                            yield from self.gen_subgraph_rules(
                                subgraph, counter)
                            counter += 1
                        # elif len(subset) == 1 and 'word' not in query_string:
                        elif (
                                (binary or len(subset) == len(subgraph_nodes))
                                and 'word' not in query_string):
                            head = graph["root"][1]
                            dep_before = query_string
                            dep_after = ""
                            yield from self.gen_rules(
                                head, dep_before, dep_after, 9000 + counter)
                            counter += 1

                        if lemma_query_string in self.subgraphs_highest:
                            subgraph = self.query_order(
                                constraints, lemma_query_string)
                            yield from self.gen_subgraph_rules(
                                subgraph, counter)
                            counter += 1
            else:
                start_rule_set = set()
                for e in graph["graph"]:
                    for element in e["to"]:
                        start_rule_set.add(element)
                yield from self.gen_start_rules(start_rule_set)

    def gen_rules(self, h, d_before, d_after, counter):
        rewrite_rule = (
            h.upper() + " -> rule_" + str(counter) + "(" + h.upper() + ",")
        if not d_before and not d_after:
            return
        # print('h, d_b, d_a:', h, d_before, d_after)
        before_nodes = []
        before_edges = []
        after_nodes = []
        after_edges = []

        if d_before:
            for n in d_before.split("&"):
                n = n.split("|")
                rewrite_rule += n[0].upper() + ","
                before_nodes.append(n[0])
                before_edges.append(n[1])

        if d_after:
            for n in d_after.split("&"):
                n = n.split("|")
                rewrite_rule += n[0].upper() + ","
                after_nodes.append(n[0])
                after_edges.append(n[1])

        rewrite_rule = rewrite_rule.strip(",")
        rewrite_rule += ") "
        rewrite_rule += "[0.99]"

        yield rewrite_rule
        yield from self.generate_string_line(h, before_nodes, after_nodes)
        yield from self.generate_graph_line(before_edges, after_edges)
        yield "\n"

    def generate_string_line(self, h, before_nodes, after_nodes):
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

        yield string_line

    def generate_graph_line(self, before_edges, after_edges):
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

        yield graph_line

    def gen_start_rules(self, s):
        for i in s:
            yield "S! -> start_b_{}({}) [1.0]".format(
                i.upper(), i.upper())

            yield "[string] ?1"
            yield "[ud] ?1"
            yield "\n"

    def gen_grammar_lines(
            self, sen_rules, sen, max_subset_size, binary=False):
        yield from self.generate_grammar(
            sen_rules, max_subset_size, binary=binary)
        yield from self.generate_terminal_ids(sen)

    def serve(self, port):
        print('launching Flask app...')
        app = Flask(__name__)

        @app.route('/get_grammar_lines', methods=['POST'])
        def get_grammar_lines():
            r = request.json
            rules = r['rules']
            assert rules
            sen = r['sen']
            binary = r['binary']
            max_subset_size = r['max_subset_size']
            grammar_lines = list(self.gen_grammar_lines(
                rules, sen, max_subset_size, binary))
            ret_value = {"grammar_lines": grammar_lines}
            return jsonify(ret_value)

        @app.route('/get_word_to_id', methods=['POST'])
        def get_word_to_id():
            ret_value = {"word_to_id": self.word_to_id}
            return jsonify(ret_value)

        app.run(port=port)


class GrammarClient():
    def __init__(self, host):
        self.host = host
        self.word_to_id = self.get_word_to_id()

    def get_word_to_id(self):
        r = requests.post(f'{self.host}/get_word_to_id')
        data = r.json()
        return data['word_to_id']

    def get_grammar_lines(self, rules, sen, max_subset_size, binary=False):
        host = f'{self.host}/get_grammar_lines'
        payload = {
            "rules": rules,
            "sen": sen,
            "max_subset_size": max_subset_size,
            "binary": binary}
        # print('sending to {host}, this: {payload}')
        r = requests.post(host, json=payload)
        data = r.json()
        return data['grammar_lines']


def get_args():
    parser = argparse.ArgumentParser(
        description="Train or load grammar and run server")
    parser.add_argument("--max_subset_size", type=int, default=5,
                        help="maximum allowed subgraph size (excl. head)")
    parser.add_argument("--model_file", type=str,
                        help="path to model file to save to or load from")
    parser.add_argument("--train_file", type=str,
                        help="path to the CoNLL train file")
    parser.add_argument("--test_file", type=str,
                        help="path to the CoNLL test file")
    parser.add_argument("--port", type=int, default=4784,
                        help="server port")
    return parser.parse_args()


def train_or_load_model(args):
    if args.train_file:
        grammar = Grammar()
        logging.info(f'training model from {args.train_file}...')
        grammar.build_dictionaries([args.train_file, args.test_file])
        grammar.train_subgraphs(args.train_file, args.max_subset_size)
        logging.info(f'saving model to {args.model_file}...')
        grammar.save(args.model_file)
    elif args.model_file:
        logging.info(f'loading model from {args.model_file}...')
        grammar = Grammar.load(args.model_file)
        logging.info('done!')
    else:
        print('no training file and no pre-trained model file provided!')
        sys.exit(-1)

    return grammar


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    args = get_args()
    grammar = train_or_load_model(args)
    grammar.serve(args.port)


if __name__ == '__main__':
    main()
