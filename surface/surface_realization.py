import argparse
import logging
import os
import subprocess

from stanza.models.common.doc import Document as StanzaDocument

from surface.grammar import GrammarClient
from surface.utils import (
    create_alto_input,
    gen_conll_sens_from_file,
    get_graph,
    get_ids_from_parse,
    get_isi_sgraph,
    get_rules,
    merge_sens,
    print_conll_sen,
    reorder_sentence,
    split_sen_on_edges,
    words
)


RECURSIVE = True

SPLIT_EDGES = {
    "acl",
    "advcl",
    "ccomp",
    "xcomp",
    "conj"
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and generate IRTG parser")
    parser.add_argument("--gen_dir", type=str,
                        help="path to save generated grammars and data files")
    parser.add_argument("--output_file", type=str,
                        help="path to output file")
    parser.add_argument("--host", type=str, default='http://localhost',
                        help="grammar server port")
    parser.add_argument("--max_subset_size", type=int, default=5,
                        help="maximum allowed subgraph size (excl. head)")
    parser.add_argument("--port", type=int, default=4784,
                        help="grammar server port")
    parser.add_argument("--test_file", type=str,
                        help="path to the CoNLL test file")
    parser.add_argument("--timeout", type=int, default=5,
                        help="default timeout for alto")
    parser.add_argument("--timeout_bin", type=int, default=60,
                        help="timeout for binary grammars")
    return parser.parse_args()


def get_alto_command(timeout, input_fn, grammar_fn, output_fn):
    return [
        'timeout', str(timeout), 'java', '-Xmx32G', '-cp',
        'alto-2.3.6-all.jar',
        'de.up.ling.irtg.script.ParsingEvaluator', input_fn,
        '-g', grammar_fn, '-I', 'ud', '-O', 'string=toString',
        '-o', output_fn]


def predict_word_order(sen, i, root_id, grammar, args):
    """expects stanza Sentence, surface.Grammar, args"""

    graph = get_graph(sen, grammar.word_to_id)
    isi_sgraph = get_isi_sgraph(graph, root_id)
    rules = get_rules(graph)

    grammar_fn, input_fn, output_fn = (
        os.path.join(args.gen_dir, fn) for fn in (
            f'{i}.irtg', f'{i}.input', f'{i}.output'))

    create_alto_input(input_fn, isi_sgraph)

    grammar_lines = grammar.get_grammar_lines(
        rules, sen, args.max_subset_size)
    with open(grammar_fn, 'w') as grammar_f:
        grammar_f.write("\n".join(grammar_lines))
    command = get_alto_command(
        args.timeout, input_fn, grammar_fn, output_fn)
    cproc = subprocess.run(command)
    if cproc.returncode == 124:
        print(f'sen {i} timed out, falling back to binary grammar')
        grammar_fn = f'{i}_bin.irtg'
        grammar_lines = grammar.get_grammar_lines(
            rules, sen, args.max_subset_size, binary=True)
        with open(grammar_fn, 'w') as grammar_f:
            grammar_f.write("\n".join(grammar_lines))
        command = get_alto_command(
            args.timeout_bin, input_fn, grammar_fn, output_fn)
        cproc = subprocess.run(command)
        if cproc.returncode == 124:
            print(f'sen {i} timed out again, skipping')
            return
    elif cproc.returncode != 0:
        print(f'alto error on sentence {i}, skipping')
        return

    try:
        pred_ids = get_ids_from_parse(output_fn)
        return pred_ids
    except IndexError:
        print(f'no parse for sentence {i}, skipping')
        return


def orig_order(toks):
    return sorted(
        toks, key=lambda tok: int(tok.feats.split('|')[-1].split('=')[-1]))


def one_step_surface_realization(
        sen, sen_id, root_id, grammar, args, keep_ids=False):
    pred_ids = predict_word_order(sen, sen_id, root_id, grammar, args)
    out_sen = reorder_sentence(sen, pred_ids, keep_ids)
    return out_sen


def rec_surface_realization(sen, sen_id, root_head, grammar, args):
    logging.debug(f'*** {sen_id} ***')
    logging.debug(words(sen))
    logging.debug('******')
    top_sen, root_id, subsens = split_sen_on_edges(sen, root_head, SPLIT_EDGES)

    if len(subsens) == 1:
        logging.debug('no real split, running single step SR')
        reordered_sen = one_step_surface_realization(
            sen, sen_id, root_id, grammar, args, keep_ids=True)
        return reordered_sen, root_id

    logging.debug('split results:')
    logging.debug('{0} {1}'.format('top_sen:', words(top_sen)))
    for i, (subsen, s_root_id) in enumerate(subsens):
        logging.debug('subsen {0}, children of #{1}: {2}'.format(
            i, s_root_id, words(subsen)))

    reordered_top_sen = one_step_surface_realization(
        top_sen, sen_id, root_id, grammar, args, keep_ids=True)

    reordered_subsens = []
    for i, (subsen, s_root_head) in enumerate(subsens):
        subsen_id = f"{sen_id}_{i}"
        reordered_subsen, s_root_id = rec_surface_realization(
            subsen, subsen_id, s_root_head, grammar, args)
        reordered_subsens.append((reordered_subsen, s_root_id))

    reordered_sen = merge_sens(reordered_top_sen, reordered_subsens)

    logging.debug('merged sen: {}'.format(words(reordered_sen)))

    return reordered_sen, root_id


def surface_realization(sen, sen_id, grammar, args):
    toks = sen.to_dict()
    if RECURSIVE:
        reordered_sen, _ = rec_surface_realization(
            toks, sen_id, 0, grammar, args)
    else:
        reordered_sen = one_step_surface_realization(
            toks, sen_id, grammar, args)

    return StanzaDocument([reordered_sen]).sentences[0]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    args = get_args()
    assert os.path.isdir(args.gen_dir)
    grammar = GrammarClient(f"{args.host}:{args.port}")
    with open(args.output_file, "w") as f:
        for sen_id, sen in enumerate(
                gen_conll_sens_from_file(args.test_file, swaps=((1, 2),))):
            print(f'processing sentence {sen_id}...')
            out_sen = surface_realization(sen, sen_id, grammar, args)

            f.write(print_conll_sen(out_sen, sen_id))
            f.write('\n')


if __name__ == "__main__":
    main()
