import argparse
import os
import pickle
import subprocess
import sys

from surface import converter
from surface import utils
from surface.grammar import Grammar


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and generate IRTG parser")
    parser.add_argument("--gen_dir", type=str,
                        help="path to save generated grammars and data files")
    parser.add_argument("--model_file", type=str,
                        help="path to model file to save to or load from")
    parser.add_argument("--output_file", type=str,
                        help="path to output file")
    parser.add_argument("--train_file", type=str,
                        help="path to the CoNLL train file")
    parser.add_argument("--test_file", type=str,
                        help="path to the CoNLL test file")
    parser.add_argument("--timeout", type=int, default=5,
                        help="default timeout for alto")
    parser.add_argument("--timeout_bin", type=int, default=60,
                        help="path to the CoNLL test file")
    return parser.parse_args()


def train_or_load_model(args):
    if args.train_file:
        word_to_id, id_to_word = converter.build_dictionaries(
            [args.train_file, args.test_file])
        grammar = Grammar()
        print(f'training model from {args.train_file}...')
        grammar.train_subgraphs(args.train_file, word_to_id)
        print(f'saving model to {args.model_file}...')
        with open('grammar.bin', 'wb') as f:
            pickle.dump({
                "model": grammar,
                "word_to_id": word_to_id,
                "id_to_word": id_to_word}, f)
        return grammar, word_to_id, id_to_word
    elif args.model_file:
        print(f'loading model from {args.model_file}...')
        with open('grammar.bin', 'rb') as f:
            d = pickle.load(f)
            grammar = d['grammar']
            word_to_id = d['word_to_id']
            id_to_word = d['id_to_word']
        return grammar, word_to_id, id_to_word
    else:
        print('no training file and no pre-trained model file provided!')
        sys.exit(-1)


def get_alto_command(timeout, input_fn, grammar_fn, output_fn):
    return [
        'timeout', str(timeout), 'java', '-Xmx32G', '-cp',
        'alto-2.3.6-all.jar',
        'de.up.ling.irtg.script.ParsingEvaluator', input_fn,
        '-g', grammar_fn, '-I', 'ud', '-O', 'string=toString',
        '-o', output_fn]


def surface_realization(grammar, word_to_id, args):
    rules, _ = converter.extract_rules(args.test_file, word_to_id)
    graphs, _, id_graphs = converter.convert(args.test_file, word_to_id)
    conll = utils.get_conll_from_file(args.test_file, word_to_id)
    pred_ids = {}
    for i in range(len(rules)):
        print(f'processing sentence {i}...')
        grammar_fn, input_fn, output_fn = (
            os.path.join(args.gen_dir, fn) for fn in (
                f'{i}.irtg', f'{i}.input', f'{i}.output'))
        utils.set_parse(input_fn, id_graphs[i])
        grammar.gen_grammar_file(rules[i], grammar_fn, conll[i])
        command = get_alto_command(
            args.timeout, input_fn, grammar_fn, output_fn)
        cproc = subprocess.run(command)
        if cproc.returncode == 124:
            print(f'sen {i} timed out, falling back to binary grammar')
            grammar.gen_grammar_file(
                rules[i], grammar_fn, conll[i], binary=True)
            command = get_alto_command(
                args.timeout_bin, input_fn, grammar_fn, output_fn)
            cproc = subprocess.run(command)
            if cproc.returncode == 124:
                print(f'sen {i} timed out again, skipping')
                pred_ids[i] = None
                continue
        elif cproc.returncode != 0:
            print(f'alto error on sentence {i}, skipping')
            pred_ids[i] = None
            continue

        try:
            pred_ids[i] = utils.get_ids_from_parse(output_fn)
        except IndexError:
            print(f'no parse for sentence {i}, skipping')
            pred_ids[i] = None

    return pred_ids, conll


def orig_order(toks):
    return sorted(
        toks, key=lambda tok: int(tok.misc.split('|')[-1].split('=')[-1]))


def print_results(pred_ids, conll, output_fn):
    with open(output_fn, "w") as f:
        for sen_id, ids in pred_ids.items():
            sen = conll[sen_id]
            words = " ".join(tok.word for tok in orig_order(sen))
            f.write(f"# {sen_id}: {words}\n")
            if ids is None:
                f.write(f"# no parse for sentence {sen_id}\n\n")
                continue

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

                f.write("\t".join(str(f) for f in new_tok))
                f.write('\n')
            f.write("\n")


def main():
    args = get_args()
    assert os.path.isdir(args.gen_dir)
    grammar, word_to_id, _ = train_or_load_model(args)
    pred_ids, conll = surface_realization(grammar, word_to_id, args)
    print_results(pred_ids, conll, args.output_file)


if __name__ == "__main__":
    main()
