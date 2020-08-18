from argparse import ArgumentParser

comment = "#"


def calculate_for_word_pos(dictionary, key, space_info, next_yes, next_no):
    if key not in dictionary:
        dictionary[key] = {"space after yes": 0, "space after no": 0, "space before yes": 0, "space before no": 0}
    if next_yes:
        dictionary[key]["space before yes"] += 1
    elif next_no:
        dictionary[key]["space before no"] += 1
    if "SpaceAfter=No" in space_info:
        dictionary[key]["space after no"] += 1
        next_yes = False
        next_no = True
    else:
        dictionary[key]["space after yes"] += 1
        next_yes = True
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


def write_prediction(predicted, previous_line, space_after):
    predicted.write("{}\t{}\n".format(previous_line, space_string(space_after)))


def check_prediction(measures, truth, predicted):
    prediction = predicted >= 0.5
    if truth == prediction and truth:
        measures["tp"] += 1
    elif truth == prediction and not truth:
        measures["tn"] += 1
    elif truth != prediction and truth:
        measures["fn"] += 1
    else:
        measures["fp"] += 1


def calculate_relevant_space(dictionary, key):
    return dictionary[key]["space after yes"] / (dictionary[key]["space after yes"] + dictionary[key]["space after no"])


def add_space_before(space_after, avg):
    space_after += avg
    return space_after / 2


def handle_word_space(same_sentence, space_after, result_file, previous_line, previous_space, measures,
                      dictionary=None, key=None, average_space_after=None):
    if (dictionary is None or key is None) and average_space_after is None:
        raise Exception("Either dictionary and key or average space is needed for the calculations!")
    if dictionary is not None and key is not None:
        avg = calculate_relevant_space(dictionary, key)
    else:
        avg = average_space_after
    if same_sentence and len(space_after) > 0:
        space_after[-1] = add_space_before(space_after[-1], avg)
        write_prediction(result_file, previous_line, space_after[-1])
        check_prediction(measures, previous_space, space_after[-1])
    space_after.append(avg)


def predict_space(dev, space_stat, space_stat_pos, save_path):
    measures = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    average_space_after = sum([calculate_relevant_space(space_stat, word) for word in space_stat]) / len(space_stat)
    with open(save_path, "w") as predicted:
        space_after = []
        previous_line = None
        previous_space = None
        with open(dev) as conll_file:
            same_sentence = True
            for line in conll_file:
                if line.startswith(comment) or line == "\n":
                    same_sentence = False
                    if previous_line is not None:
                        write_prediction(predicted, previous_line, space_after[-1])
                        check_prediction(measures, previous_space, space_after[-1])
                    predicted.write(line)
                    continue
                word = line.split("\t")[1]
                pos = line.split("\t")[3]
                if word not in space_stat:
                    if pos not in space_stat_pos:
                        handle_word_space(same_sentence, space_after, predicted, previous_line, previous_space,
                                          measures, average_space_after=average_space_after)
                    else:
                        handle_word_space(same_sentence, space_after, predicted, previous_line, previous_space,
                                          measures, space_stat_pos, pos)
                else:
                    handle_word_space(same_sentence, space_after, predicted, previous_line, previous_space,
                                      measures, space_stat, word)
                same_sentence = True
                previous_line = "\t".join(line.split("\t")[:-1])
                previous_space = "SpaceAfter=No" not in line.split("\t")[-1]
    measures["precision"] = measures["tp"] / (measures["tp"] + measures["fp"])
    measures["recall"] = measures["tp"] / (measures["tp"] + measures["fn"])
    measures["f1"] = 2 * (measures["precision"] * measures["recall"]) / (measures["precision"] + measures["recall"])
    return measures


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-t", "--train", nargs='+', required=True)
    args.add_argument("-d", "--dev", nargs='+', required=True)
    args.add_argument("-o", "--out", nargs='+', default="output.conll")
    arguments = args.parse_args()
    statistics = {}
    for t, d, o in zip(arguments.train, arguments.dev, arguments.out):
        space_stats, space_stats_pos = calculate_space_statistics(t)
        statistics[o] = predict_space(d, space_stats, space_stats_pos, o)
    with open("statistics.csv", "w") as stats_file:
        stats_file.write("File\tTP\tFP\tTN\tFN\tPrecision\tRecall\tF1\n")
        for data in statistics.items():
            stats_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0], data[1]["tp"], data[1]["fp"],
                                                                       data[1]["tn"], data[1]["fn"],
                                                                       data[1]["precision"], data[1]["recall"],
                                                                       data[1]["f1"]))
