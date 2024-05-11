from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
# from bert_score import score
from statistics import mean
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from rouge import Rouge
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import math

def calc_bleu(cands, refs, print_score: bool = True):
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    for c, r in zip(cands, refs):
        r = [word_tokenize(r)]
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
        ref = word_tokenize(r)
        hypo = word_tokenize(p)
        ms = round(meteor_score([ref], hypo), 4)
        scores.append(ms)
    final_score = mean(scores)
    if print_score:
        print(f"***** METEOR Score: {final_score*100} *****")
    return final_score


def calc_bertscore(cands, refs, print_score: bool = True):
    # p, r, f1 = score(cands, refs, lang="en", verbose=False, rescale_with_baseline=True)
    # scores = f1.detach().numpy()
    # final_score = mean(scores)
    hf_bertscorer = load("bertscore")
    final_score = mean(hf_bertscorer.compute(predictions=cands, rs=refs, lang='en', rescale_with_baseline=True)['f1'])
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


def cal_vector_extrema(x, y, dic):
    # x and y are the list of the words
    # dic is the gensim model which holds 300 the google news word2ved model
    def vecterize(p):
        vectors = []
        for w in p:
            if w.lower() in dic:
                vectors.append(dic[w.lower()])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    vec_x = np.max(x, axis=0)
    vec_y = np.max(y, axis=0)
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    zero_list = np.zeros(len(vec_x))
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos


def cal_embedding_average(x, y, dic):
    # x and y are the list of the words
    def vecterize(p):
        vectors = []
        for w in p:
            if w.lower() in dic:
                vectors.append(dic[w.lower()])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)

    vec_x = np.array([0 for _ in range(len(x[0]))])
    for x_v in x:
        x_v = np.array(x_v)
        vec_x = np.add(x_v, vec_x)
    vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))

    vec_y = np.array([0 for _ in range(len(y[0]))])
    #print(len(vec_y))
    for y_v in y:
        y_v = np.array(y_v)
        vec_y = np.add(y_v, vec_y)
    vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))

    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"

    zero_list = np.array([0 for _ in range(len(vec_x))])
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)

    vec_x = np.mat(vec_x)
    vec_y = np.mat(vec_y)
    num = float(vec_x * vec_y.T)
    denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
    cos = num / denom

    # res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos



def read_file(file_name, dec_type="Greedy"):
    f = open(file_name, "r", encoding="utf-8")

    refs = []
    cands = []
    dec_str = f"{dec_type}: "

    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
            # EVAL	Loss: 3.2405	PPL: 25.5458	BCE: 2.8742	Accuracy: 0.3302 0.8001 0.5537
            # f.write("Loss: {:.4f}\tPPL: {:.4f}\tBCE: {:.4f}\tAccuracy: {:.4f} {:.4f} {:.4f}\n"
            _, ppl, _, acc = line.strip("\n").split('\t')
            ppl = ppl.split(': ')[1]
            acc = acc.split(': ')[1].split(' ')

            # print(f"PPL: {ppl}\tEmotion Accuracy: {float(acc[0])*100}%, {float(acc[1])*100}%\tAct Accuracy: {float(acc[2])*100}%")
            # print("PPL: {}\tEmotion Accuracy: {:.2f}%, {:.2f}%\tAct Accuracy: {:.2f}%".format(ppl, float(acc[0])*100, float(acc[1])*100, float(acc[2])*100))
            print("PPL: {}\tEmotion Accuracy: {:.2f}%\tPol Accuracy: {:.2f}%".format(ppl, float(acc[0])*100, float(acc[1])*100))

        if line.startswith(dec_str):
            exp = line.strip(dec_str).strip("\n")
            cands.append(exp)
        if line.startswith("Ref: "):
            ref = line.strip("Ref: ").strip("\n")
            refs.append(ref)

    # return refs, cands, float(ppl), float(acc[0])
    return refs, cands, 0, 0


if __name__ == "__main__":
    files = [
        'output/eqt/results.txt'
    ]

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
        eval_with_evaluator(cands, refs)

    # print(ppl, acc, d1, d2, bert_score, meteor)
    # print(ppl, acc, d1, d2)
    print(ppl, acc, d1, d2)
