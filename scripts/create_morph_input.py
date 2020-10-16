import sys

from surface.utils import gen_conll_sens, print_conll_sen


def main():
    for sen in gen_conll_sens(sys.stdin):
        for tok in sen.words:
            orig_id = f'original_id={tok.id}'
            tok.feats = tok.feats + f'|{orig_id}' if tok.feats else orig_id
        print(print_conll_sen(sen))


if __name__ == '__main__':
    main()
