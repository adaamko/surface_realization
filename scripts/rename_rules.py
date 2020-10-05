import re
import sys

patt1 = re.compile(r'^.* rule_([0-9]*)\(.*$')
patt2 = re.compile(r'^.* pos_to_word_([0-9]*)\(.*$')

counts = [0, 0]
for line in sys.stdin:
    for i, patt in enumerate((patt1, patt2)):
        m = patt.match(line)
        if m is None:
            continue
        else:
            sys.stdout.write(line.replace(m.group(1), f'{counts[i]}'))
            counts[i] += 1
            break
    else:
        sys.stdout.write(line)
