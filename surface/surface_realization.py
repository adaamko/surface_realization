import argparse
import os
import subprocess

from surface import converter
from surface import utils
from surface.grammar import GrammarClient


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


def surface_realization(grammar, args):
    rules, _ = converter.extract_rules(args.test_file, grammar.word_to_id)
    graphs, _, id_graphs = converter.convert(
        args.test_file, grammar.word_to_id)
    conll = utils.get_conll_from_file(
        args.test_file, grammar.word_to_id)
    for i in range(len(rules)):
        print(f'processing sentence {i}...')
        grammar_fn, input_fn, output_fn = (
            os.path.join(args.gen_dir, fn) for fn in (
                f'{i}.irtg', f'{i}.input', f'{i}.output'))
        utils.set_parse(input_fn, id_graphs[i])
        grammar_lines = grammar.get_grammar_lines(
            rules[i], conll[i], args.max_subset_size)
        with open(grammar_fn, 'w') as grammar_f:
            grammar_f.write("\n".join(grammar_lines))
        command = get_alto_command(
            args.timeout, input_fn, grammar_fn, output_fn)
        cproc = subprocess.run(command)
        if cproc.returncode == 124:
            print(f'sen {i} timed out, falling back to binary grammar')
            grammar_fn = f'{i}_bin.irtg'
            grammar_lines = grammar.get_grammar_lines(
                rules[i], conll[i], args.max_subset_size, binary=True)
            with open(grammar_fn, 'w') as grammar_f:
                grammar_f.write("\n".join(grammar_lines))
            command = get_alto_command(
                args.timeout_bin, input_fn, grammar_fn, output_fn)
            cproc = subprocess.run(command)
            if cproc.returncode == 124:
                print(f'sen {i} timed out again, skipping')
                yield conll[i], None
                continue
        elif cproc.returncode != 0:
            print(f'alto error on sentence {i}, skipping')
            yield conll[i], None
            continue

        try:
            pred_ids = utils.get_ids_from_parse(output_fn)
            yield conll[i], pred_ids
        except IndexError:
            print(f'no parse for sentence {i}, skipping')
            yield conll[i], None


def orig_order(toks):
    return sorted(
        toks, key=lambda tok: int(tok.misc.split('|')[-1].split('=')[-1]))


def gen_result_lines(sen_id, sen, ids):
    words = " ".join(tok.word for tok in orig_order(sen))
    yield f"# {sen_id}: {words}"
    if ids is None:
        yield f"# no parse for sentence {sen_id}"
        yield "\n"
        return

    old_id_to_tok = {tok.id: tok for tok in sen}
    tok_to_new_id = {tok: ids.index(tok.id) for tok in sen}
    for new_id, tok in enumerate(
            sorted(sen, key=lambda t: tok_to_new_id[t])):
        if tok.head == 0:
            head_id = 0
        else:
            head_tok = old_id_to_tok[tok.head]
            head_id = tok_to_new_id[head_tok] + 1

        new_tok = utils.Token(
            new_id+1, tok.lemma, tok.word, tok.pos, tok.tpos, tok.misc,
            head_id, tok.deprel, tok.comp_edge, tok.space_after,
            tok.word_id)

        yield "\t".join(str(f) for f in new_tok)
    yield "\n"


def main():
    args = get_args()
    assert os.path.isdir(args.gen_dir)
    grammar = GrammarClient(f"{args.host}:{args.port}")
    with open(args.output_file, "w") as f:
        for sen_id, (sen, pred_ids) in enumerate(
                surface_realization(grammar, args)):
            f.write("\n".join(gen_result_lines(sen_id, sen, pred_ids)))


if __name__ == "__main__":
    main()
