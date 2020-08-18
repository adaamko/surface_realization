from argparse import ArgumentParser

comment = "#"


def calculate_for_word_pos(dictionary, key, space_info, next_yes, next_no):
    if key not in dictionary:
        dictionary[key] = {"space after yes": 0, "space after no": 0, "space before yes": 0, "space before no": 0}
    if next_yes:
        dictionary[key]["space before yes"] += 1
        next_yes = False
    elif next_no:
        dictionary[key]["space before no"] += 1
        next_no = False
    if space_info == "_":
        dictionary[key]["space after yes"] += 1
        next_yes = True
    elif space_info == "SpaceAfter=No":
        dictionary[key]["space after no"] += 1
        next_no = True
    else:
        next_yes = False
        next_no = False
    return next_yes, next_no


def calculate_space_statistics(train):
    space_stat = {}
    space_stat_pos = {}
    with open(train) as conll_file:
        next_yes = False
        next_no = False
        for line in conll_file:
            if line.startswith(comment) or line == "\n":
                next_yes = False
                next_no = False
                continue
            space_info = line.split("\t")[-1].strip()
            word = line.split("\t")[1]
            pos = line.split("\t")[3]
            calculate_for_word_pos(space_stat, word, space_info, next_yes, next_no)
            next_yes, next_no = calculate_for_word_pos(space_stat_pos, pos, space_info, next_yes, next_no)
    return space_stat, space_stat_pos


def space_string(space_after):
    return "_" if space_after >= 0.5 else "SpaceAfter=No"


def predict_space(dev, space_stat, space_stat_pos, save_path):
    average_space_after = sum([space_stat[word]["space after yes"]/(space_stat[word]["space after yes"] +
                                                                    space_stat[word]["space after no"])
                               for word in space_stat])/len(space_stat)
    with open(save_path, "w") as predicted:
        space_after = []
        previous_line = None
        with open(dev) as conll_file:
            same_sentence = True
            for line in conll_file:
                if line.startswith(comment) or line == "\n":
                    same_sentence = False
                    predicted.write("{}\t{}\n".format(previous_line, space_string(space_after[-1])))
                    predicted.write(line)
                    continue
                word = line.split("\t")[1]
                pos = line.split("\t")[3]
                if word not in space_stat:
                    if pos not in space_stat_pos:
                        if same_sentence and len(space_after) > 0:
                            space_after[-1] += average_space_after
                            space_after[-1] /= 2
                            predicted.write("{}\t{}\n".format(previous_line, space_string(space_after[-1])))
                        space_after.append(average_space_after)
                    else:
                        pos_avg = space_stat_pos[pos]["space after yes"]/(space_stat_pos[pos]["space after yes"] +
                                                                          space_stat_pos[pos]["space after no"])
                        if same_sentence and len(space_after) > 0:
                            space_after[-1] += pos_avg
                            space_after[-1] /= 2
                            predicted.write("{}\t{}\n".format(previous_line, space_string(space_after[-1])))
                        space_after.append(pos_avg)
                else:
                    word_avg = space_stat[word]["space after yes"]/(space_stat[word]["space after yes"] +
                                                                    space_stat[word]["space after no"])
                    if same_sentence and len(space_after) > 0:
                        space_after[-1] += word_avg
                        space_after[-1] /= 2
                        predicted.write("{}\t{}\n".format(previous_line, space_string(space_after[-1])))
                    space_after.append(word_avg)
                same_sentence = True
                previous_line = "\t".join(line.split("\t")[:-1])


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-t", "--train", required=True)
    args.add_argument("-d", "--dev", required=True)
    args.add_argument("-o", "--out", default="output.conll")
    arguments = args.parse_args()
    space_stats, space_stats_pos = calculate_space_statistics(arguments.train)
    predict_space(arguments.dev, space_stats, space_stats_pos, arguments.out)
