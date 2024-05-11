import json

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from statistics import mean
from nltk.translate.bleu_score import sentence_bleu


def calc_bleu(cands, refs, print_score: bool = True):
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    for c, r in zip(cands, refs):
        # r = [word_tokenize(r)]
        r = [word_tokenize(r_i) for r_i in r]
        c = word_tokenize(c)
        bleu1.append(sentence_bleu(r, c, weights=(1, 0, 0, 0)))
        bleu2.append(sentence_bleu(r, c, weights=(0, 1, 0, 0)))
        bleu3.append(sentence_bleu(r, c, weights=(0, 0, 1, 0)))
        bleu4.append(sentence_bleu(r, c, weights=(0, 0, 0, 1)))
    b1, b2, b3, b4 = mean(bleu1), mean(bleu2), mean(bleu3), mean(bleu4)
    if print_score:
        print(f"***** BLEU-1: {b1*100} *****")
        print(f"***** BLEU-2: {b2 * 100} *****")
        print(f"***** BLEU-3: {b3 * 100} *****")
        print(f"***** BLEU-4: {b4 * 100} *****")
    return b1, b2, b3, b4


def calc_meteor(cands, refs, print_score: bool = True):
    scores = []
    for p, r in zip(cands, refs):
        ref = [word_tokenize(r_i) for r_i in r]
        hypo = word_tokenize(p)
        # ms = round(meteor_score([ref], hypo), 4)
        ms = round(meteor_score(ref, hypo), 4)
        scores.append(ms)
    final_score = mean(scores)
    if print_score:
        print(f"***** METEOR Score: {final_score*100} *****")
    return final_score


def calc_bertscore(cands, refs, print_score: bool = True):
    p, r, f1 = score(cands, refs, lang="en", verbose=False, rescale_with_baseline=True)
    scores = f1.detach().numpy()
    final_score = mean(scores)
    if print_score:
        print(f"***** Bert Score: {final_score*100} *****")
    return  final_score


def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score


def calc_distinct(candidates, print_score: bool = True):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores


def read_file(file_name, dec_type="Greedy"):
    # f = open(f"results/{file_name}.txt", "r", encoding="utf-8")
    # f = open(f"save/{file_name}/results.txt", "r", encoding="utf-8")
    f = open(file_name, "r", encoding="utf-8")

    refs = []
    cands = []
    dec_str = f"{dec_type}: "

    cand_ref = []
    for i, line in enumerate(f.readlines()):
        if i == 0:
            # EVAL	Loss: 3.2405	PPL: 25.5458	BCE: 2.8742	Accuracy: 0.3302 0.8001 0.5537
            # f.write("Loss: {:.4f}\tPPL: {:.4f}\tBCE: {:.4f}\tAccuracy: {:.4f} {:.4f} {:.4f}\n"
            _, ppl, _, acc = line.strip("\n").split('\t')
            ppl = ppl.split(': ')[1]
            acc = acc.split(': ')[1].split(' ')

            # print(f"PPL: {ppl}\tEmotion Accuracy: {float(acc[0])*100}%, {float(acc[1])*100}%\tAct Accuracy: {float(acc[2])*100}%")
            # print("PPL: {}\tEmotion Accuracy: {:.2f}%, {:.2f}%\tAct Accuracy: {:.2f}%".format(ppl, float(acc[0])*100, float(acc[1])*100, float(acc[2])*100))
            if len(acc) == 3:
                print("PPL: {}\tEmotion Accuracy: {:.2f}%, {:.2f}%\tAct Accuracy: {:.2f}%".format(ppl, float(acc[0])*100, float(acc[1])*100, float(acc[2])*100))
            else:
                print("PPL: {}\tEmotion Accuracy: {:.2f}%\tPol Accuracy: {:.2f}%".format(ppl, float(acc[0])*100, float(acc[1])*100))

        if line.startswith(dec_str):
            exp = line.strip(dec_str).strip("\n")
            cands.append(exp)
        if line.startswith("Ref: "):
            ref = line.strip("Ref: ").strip("\n")
            # refs.append(ref)
            cand_ref.append(ref)
        if line.startswith('----------'):
            refs.append(cand_ref.copy())
            cand_ref = []

    assert len(refs) == len(cands), 'Invalid Data Size: Refs {}, Cands {}'.format(len(refs), len(cands))

    return refs, cands, float(ppl), float(acc[0])


def eval_metrics(files):
    best_ppl = 50
    best_acc = 0
    best_dist1 = 0
    best_dist2 = 0
    ppl = ""
    acc = ""
    d1 = ""
    d2 = ""
    bert_score = 0
    meteor = 0
    b1, b2, b3, b4 = 0, 0, 0, 0
    for f in files:
        print(f"Evaluating {f}")
        refs, cands, p, a = read_file(f)
        if p < best_ppl:
            ppl = f
            best_ppl = p
        if a > best_acc:
            acc = f
            best_acc = a
        dist_1, dist_2 = calc_distinct(cands)
        if dist_1 > best_dist1:
            d1 = f
            best_dist1 = dist_1
        if dist_2 > best_dist2:
            d2 = f
            best_dist2 = dist_2
        # bert_score = calc_bertscore(cands, refs)
        meteor = calc_meteor(cands, refs)
        b1, b2, b3, b4 = calc_bleu(cands, refs)
        print()

    # print(ppl, acc, d1, d2, bert_score, meteor)
    # print(ppl, acc, d1, d2)
    print(ppl, acc, d1, d2)


def get_multi_file(result_file, origin_file):
    f = open(result_file, "r", encoding="utf-8")
    output_f = open(result_file[:-4] + '_multi.txt', 'a+', encoding='utf-8')
    with open(origin_file, 'r', encoding='utf-8') as json_f:
        data = json.load(json_f)

    cur = -1
    for i, line in enumerate(f.readlines()):
        if i == 0:
            output_f.write(line)
        if line.startswith('Context:'):
            cur += 1
            output_f.write(line)
        if line.startswith('Greedy:'):
            output_f.write(line)
        if line.startswith('Ref:'):
            refs = data[cur]['reply']
            for ref in refs:
                output_f.write('Ref: {}\n'.format(ref))
        if line.startswith('----------'):
            output_f.write(line)

    assert cur + 1 == len(data), 'Invalid data size: data {}, cur {}'.format(len(data), cur + 1)

    output_f.close()
    f.close()


if __name__ == "__main__":
    files = [
        # "cem_eqt_test",
        # "empdg_eqt_test",
        # "mime_eqt_test",
        # "moel_eqt_test",
        # "multi-trs_eqt_test",
        # "trs_eqt_test",
        # "output/main_v4/results.txt"
        # "output/sample_v4/results.txt"
        "output/test2/results.txt"
        # "output/test_act_v6/results.txt"
    ]

    results = 'output/sota/results.txt'
    origin_file = 'EQ-EMAC/merge.json'
    # get_multi_file(results, origin_file)
    eval_metrics(['output/sota/results_multi.txt'])

