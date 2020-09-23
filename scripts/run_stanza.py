import sys

import stanza
from stanza.utils.conll import CoNLL
from tqdm import tqdm

nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')
for fn in tqdm(sys.argv[1:]):
    with open(fn) as f:
        text = f.read()
        doc = nlp(text)

    dic = doc.to_dict()
    conll = CoNLL.convert_dict(dic)

    with open(f"{fn}.parsed.conll", 'w') as out:
        for sen in conll:
            for tok in sen:
                out.write('\t'.join(tok))
                out.write('\n')
            out.write('\n')
