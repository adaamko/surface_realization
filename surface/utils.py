import logging
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
    for tok in sen:
        graph[tok['id']].update({
            "word": word_to_id[tok['lemma'].lower()],
            # calling upos tree_pos for backward compatibility
            "tree_pos": sanitize_word(tok['upos']),
            "mor": tok['feats']})

        graph[tok['head']]['deps'][tok['id']] = tok['deprel']

    return graph


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


def reorder_sentence(sen, ids, keep_ids):
    if ids is None:
        return sen
    new_ids = {tok['id']: ids.index(tok['id']) + 1 for tok in sen}
    toks = sorted(sen, key=lambda t: new_ids[t['id']])
    if not keep_ids:
        for tok in toks:
            tok['id'] = new_ids[tok['id']]
            if tok['head'] != 0:
                tok['head'] = new_ids[tok['head']]

    return toks


def get_subsen(toks, root):
    """works with tok dicts"""
    ids = set([root['id']])
    while True:
        n = len(ids)
        for tok in toks:
            if tok['id'] in ids:
                continue
            if tok['head'] in ids:
                ids.add(tok['id'])
        if len(ids) == n:
            break

    return [tok for tok in toks if tok['id'] in ids]


def split_sen_on_edges(toks, root_head, edges):
    root_toks = [tok for tok in toks if tok['head'] == root_head]
    assert len(root_toks) == 1, root_toks
    root_tok = root_toks[0]
    root_id = root_tok['id']

    children = [tok for tok in toks if tok['head'] == root_id]
    top_sen = [root_tok]
    subsens = []
    main_subsen = [root_tok]
    for child in children:
        subsen = get_subsen(toks, child)
        if child['deprel'] in edges:
            subsens.append((subsen, root_id))
            top_sen.append(child)
        else:
            main_subsen += subsen
    subsens.append((main_subsen, root_head))

    return top_sen, root_id, subsens


def merge_sens(reordered_top_sen, reordered_subsens):
    logging.debug('***\nMERGING THESE:\ntop sen: {0}\nids: {1}'.format(
        words(reordered_top_sen),
        [word['id'] for word in reordered_top_sen]))
    logging.debug('subsens:\n{0}\n***'.format('\n'.join(
        'root: {0}, sen: {1}'.format(root, words(ss))
        for ss, root in reordered_subsens)))

    toks = []
    ids_to_subsens = {
        s_root_id: subsen for subsen, s_root_id in reordered_subsens}

    for tok in reordered_top_sen:
        if tok['id'] in ids_to_subsens:
            toks += ids_to_subsens[tok['id']]
        else:
            toks.append(tok)

    return toks


def words(sen):
    return " ".join(tok['text'] for tok in sen)


def test():
    for sen in gen_conll_sens(sys.stdin, swaps=((1, 2),)):
        print(print_conll_sen(sen))


if __name__ == '__main__':
    test()
