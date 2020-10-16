import sys
from utils import get_sens


def eval_sen(sen):
    orig_ids = [int(tok[5].split('|')[-1].split('=')[-1]) for tok in sen]
    print("# " + " ".join(tok[2] for tok in sorted(
        sen, key=lambda tok: orig_ids[int(tok[0]) - 1])))
    print("# " + " ".join(tok[2] for tok in sen))
    for i, tok in enumerate(sen):
        # word, lemma, pos = tok[2], tok[1], tok[3]
        word, pos = tok[2], tok[3]
        dep = tok[7]
        head_id = int(tok[6])
        if head_id == 0:
            assert dep == 'root'
            continue
        htok = sen[head_id - 1]
        # hword, hlemma, hpos = htok[2], htok[1], htok[3]
        hword, hpos = htok[2], htok[3]

        if (i < head_id) != (orig_ids[i] < orig_ids[head_id - 1]):
            # print("\t".join(tok))
            # print(f"{hword}_{hlemma}_{hpos} -{dep}> {word}_{lemma}_{pos}")
            print(f"{hword}\t{hpos}\t{dep}\t{word}\t{pos}")
            # print('========================')


def main():
    for sen in get_sens(sys.stdin):
        eval_sen(sen)


if __name__ == "__main__":
    main()
