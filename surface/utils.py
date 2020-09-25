import re
import sys

from collections import defaultdict
from itertools import chain, combinations

from stanza.models.common.doc import Document as StanzaDocument
from stanza.utils.conll import CoNLL


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
NON_ENGLISH_CHARACTERS = re.compile(r"[^a-zA-Z]")

KEYWORDS = set(["feature"])

TEMPLATE = (
    '{0} -> {1}_{0}\n' +
    '[string] {1}\n' +
    '[tree] {0}({1})\n' +
    '[ud] "({1}<root> / {1})"\n' +
    '[fourlang] "({1}<root> / {1})"\n'
)


def sanitize_word(word):
    for pattern, target in REPLACE_MAP.items():
        word = word.replace(pattern, target)
    for digit in "0123456789":
        word = word.replace(digit, "DIGIT")
    if word in KEYWORDS:
        word = word.upper()
    NON_ENGLISH_CHARACTERS.sub("SPECIALCHAR", word)

    return word


def get_ids_from_parse(fn):
    with open(fn, "r") as f:
        next(f)
        parse = next(f).strip()
        return [
            int(n.strip().split('_')[1]) for n in parse.strip("[]").split(",")]


def create_alto_input(fn, graph):
    with open(fn, "w+") as f:
        f.write("# IRTG unannotated corpus file, v1.0\n")
        f.write(
            "# interpretation ud: de.up.ling.irtg.algebra.graph.GraphAlgebra"
            "\n")
        f.write(graph + "\n")
        f.write("(dummy_0 / dummy_0)\n")


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


def gen_tsv_sens(stream, swaps=()):
    curr_sen = []
    for raw_line in stream:
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if not line:
            yield curr_sen
            curr_sen = []
            continue
        fields = line.split('\t')
        for i, j in swaps:
            fields[i], fields[j] = fields[j], fields[i]
        curr_sen.append(fields)


def gen_conll_sens(stream, swaps=()):
    for sen in gen_tsv_sens(stream, swaps):
        dic = CoNLL.convert_conll([sen])
        yield StanzaDocument(dic).sentences[0]


def gen_conll_sens_from_file(fn, swaps=()):
    with open(fn, "r") as f:
        yield from gen_conll_sens(f, swaps)


def print_conll_sen(sen, sent_id=None):
    out = f'# sent_id = {sent_id}\n# text = {sen.text}\n'
    for fields in CoNLL.convert_dict([sen.to_dict()])[0]:
        out += "\t".join(fields) + '\n'
    return out


def get_graph(sen, word_to_id):
    graph = defaultdict(lambda: {"mor": "", "deps": {}})
    for tok in sen.words:
        graph[tok.id].update({
            "word": word_to_id[tok.lemma.lower()],
            # calling upos tree_pos for backward compatibility
            "tree_pos": sanitize_word(tok.upos),
            "mor": tok.feats})

        graph[tok.head]['deps'][tok.id] = tok.deprel

        if tok.head == 0:
            root_id = tok.id

    return graph, root_id


def get_isi_sgraph(graph, word_id):
    graph_string = "({1}_{0} / {1}_{0}".format(
        str(word_id), graph[word_id]["word"])
    for other_id in graph[word_id]["deps"]:
        edge = graph[word_id]["deps"][other_id]
        graph_string += ' :{0} '.format(edge.replace(':', '_'))
        graph_string += get_isi_sgraph(graph, other_id)
    graph_string += ")"
    return graph_string


def get_rules(graph):
    rules = []
    for w in graph:
        subgraphs = {"root": None, "graph": []}
        if "tree_pos" not in graph[w]:
            subgraphs["root"] = "ROOT"
            for dep in graph[w]["deps"]:
                to_pos = graph[dep]["tree_pos"]
                word = graph[dep]["word"].lower()
                subgraphs["graph"].append({
                    "to": (word.lower(), to_pos),
                    "edge": "root",
                    "dir": None})
            rules.append(subgraphs)
            continue

        subgraphs["root"] = (
            graph[w]["word"].lower(), graph[w]["tree_pos"])

        for dep in graph[w]["deps"]:
            edge_dep = graph[w]["deps"][dep]
            to_pos = graph[dep]["tree_pos"]
            word = graph[dep]["word"].lower()
            mor = graph[dep]["mor"]
            if "tree_pos" in graph[w]:
                lin_dir = (
                    "S" if "lin=+" in mor else (
                        "B" if "lin=-" in mor else None))

                subgraphs["graph"].append({
                    "to": (word.lower(), to_pos),
                    "edge": edge_dep.replace(":", "_"),
                    "dir": lin_dir})

        rules.append(subgraphs)
    return rules


def reorder_sentence(sen, ids):
    if ids is None:
        return sen
    new_ids = {tok.id: ids.index(tok.id) + 1 for tok in sen.words}
    tok_dicts = sen.to_dict()
    for tok in tok_dicts:
        tok['id'] = new_ids[tok['id']]
        if tok['head'] != 0:
            tok['head'] = new_ids[tok['head']]

    tok_dicts.sort(key=lambda t: t['id'])
    doc = StanzaDocument([tok_dicts])
    return doc.sentences[0]


def test():
    for sen in gen_conll_sens(sys.stdin, swaps=((1, 2),)):
        print(print_conll_sen(sen))


if __name__ == '__main__':
    test()
