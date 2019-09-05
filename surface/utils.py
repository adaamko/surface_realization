import os

def get_parse(fn, conll):
    with open(fn, "r") as f:
        next(f)
        conll_parse = {}
        parse = next(f).strip()
        text = [n.strip() for n in parse.strip("[]").split(",")]
        text_parse = []
        for i, w_id in enumerate(text):
            conll_parse[i] = conll[w_id.split("_")[1]]
            text_parse.append(conll[w_id.split("_")[1]][1])
        return text_parse, conll_parse
    

def set_parse(fn, graph):
    with open(fn, "w+") as f:
        f.write("# IRTG unannotated corpus file, v1.0\n")
        f.write("# interpretation ud: de.up.ling.irtg.algebra.graph.GraphAlgebra\n")
        f.write(graph + "\n")
        f.write("(dummy_0 / dummy_0)\n")