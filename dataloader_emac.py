import logging
import os
import pickle
import nltk
import torch
import torch.utils.data as data
import numpy as np
from tqdm.auto import tqdm
from comet import Comet
from constants import *
from tkinter import _flatten

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Dataset(data.Dataset):
    def __init__(self, args, data, vocab):
        self.args = args
        self.vocab = vocab
        self.data = data
        self.emo_map = EMO_MAP

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        item = dict()
        item['context_text'] = self.data['context'][index]
        item['target_text'] = self.data['target'][index]
        item['emotion_text'] = self.data['emotion'][index]
        item['polarity_text'] = POLAR_MAP[item['emotion_text']]

        item['context'], item['speaker_mask'] = self.preprocess(item['context_text'])
        item['target'] = self.preprocess(item['target_text'], tg=True)
        item['emotion'], item['emotion_label'] = self.preprocess_emo(item['emotion_text'])
        item['polarity'], item['polarity_label'] = self.preprocess_pol(item['polarity_text'])

        # item['pol_cs'] = self.data['pol_cs'][index]
        for i, r in enumerate(RELATIONS):
            r_text = "{}_text".format(r)
            item[r_text] = self.data['utt_cs'][index][i]
            item[r] = self.preprocess(item[r_text], csk=r)

        return item

    def preprocess(self, arr, tg=False, csk=None):
        if tg:
            sequence = [self.vocab.word2idx[word] if word in self.vocab.word2idx else MAP_VOCAB['UNK'] for word in arr]
            sequence.append(MAP_VOCAB['EOS'])
            return torch.LongTensor(sequence)
        if csk:
            sequence = [MAP_VOCAB['CLS']] if csk not in ['xReact', 'xAttr'] else []
            for seq in arr:
                sequence += [
                    self.vocab.word2idx[word]
                    for word in seq
                    if word in self.vocab.word2idx and word not in ["to", "none"]
                ]
            return torch.LongTensor(sequence)
        else:
            conv_token = [MAP_VOCAB['CLS']]
            speak_mask = [MAP_VOCAB['CLS']]
            for i, utt in enumerate(arr):
                conv_token += [self.vocab.word2idx[word] if word in self.vocab.word2idx else MAP_VOCAB['UNK'] for word
                               in utt]
                spk = MAP_VOCAB['USR'] if i % 2 == 0 else MAP_VOCAB['SYS']
                speak_mask += [spk for _ in range(len(utt))]
            assert len(conv_token) == len(speak_mask)
            return torch.LongTensor(conv_token), torch.LongTensor(speak_mask)

    def preprocess_emo(self, emotion):
        onehot = [0] * len(self.emo_map)
        onehot[self.emo_map[emotion]] = 1
        return onehot, self.emo_map[emotion]

    def preprocess_pol(self, polarity):
        onehot = [0] * 2
        label = None
        if polarity == 'positive':
            onehot[0] = 1
            label = 0
        elif polarity == 'negative':
            onehot[1] = 1
            label = 1
        return onehot, label

    def collate_fn(self, data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        data.sort(key=lambda x: len(x["context"]), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        input_batch, input_lengths = merge(item_info['context'])
        input_masks, input_mask_lengths = merge(item_info['speaker_mask'])
        target_batch, target_lengths = merge(item_info['target'])

        d = dict()
        d['input_batch'] = input_batch.to(self.args.device)
        d['input_masks'] = input_masks.to(self.args.device)
        d['target_batch'] = target_batch.to(self.args.device)

        d['input_lengths'] = torch.LongTensor(input_lengths)
        d['target_lengths'] = torch.LongTensor(target_lengths)

        d['target_emotion'] = item_info['emotion']
        d['emotion_label'] = item_info['emotion_label']
        d['target_polarity'] = item_info['emotion']
        d['polarity_label'] = item_info['polarity_label']
        d['input_text'] = item_info['context_text']
        d['target_text'] = item_info['target_text']
        d['emotion_text'] = item_info['emotion_text']
        d['polarity_text'] = item_info['polarity_text']

        for r in RELATIONS:
            pad_batch, _ = merge(item_info[r])
            pad_batch = pad_batch.to(self.args.device)
            d[r] = pad_batch
            d["{}_text".format(r)] = item_info["{}_text".format(r)]

        return d


class Lang:
    def __init__(self, init_vocab):
        self.idx2word = init_vocab
        self.word2idx = {str(v): int(k) for k, v in init_vocab.items()}
        self.word2count = {str(v): 1 for k, v in init_vocab.items()}
        self.n_words = len(init_vocab)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2idx:
            self.idx2word[self.n_words] = word
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1


def prepare_data_seq(args, batch_size=32):
    tra_data, val_data, tst_data, vocab = load_dataset(args)
    logger.info("Vocab Size: {}".format(vocab.n_words))

    tra_set = Dataset(args, tra_data, vocab)
    tra_loader = data.DataLoader(dataset=tra_set, batch_size=batch_size, shuffle=True, collate_fn=tra_set.collate_fn)
    val_set = Dataset(args, val_data, vocab)
    val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, collate_fn=val_set.collate_fn)
    tst_set = Dataset(args, tst_data, vocab)
    tst_loader = data.DataLoader(dataset=tst_set, batch_size=1, shuffle=False, collate_fn=tst_set.collate_fn)

    return tra_loader, val_loader, tst_loader, vocab, len(tra_set.emo_map)


def load_dataset(args):
    saved_file = os.path.join(args.data_dir, "dataset_preproc.p")
    if os.path.exists(saved_file):
        logger.info("Loading data from {}.".format(saved_file))
        with open(saved_file, "rb") as f:
            [tra_data, val_data, tst_data, vocab] = pickle.load(f)
    else:
        logger.info("Building dataset from {}.".format(args.data_dir))
        files = DATA_FILES(args.data_dir)
        tra_files = [np.load(f, allow_pickle=True) for f in files['train']]
        val_files = [np.load(f, allow_pickle=True) for f in files['dev']]
        tst_files = [np.load(f, allow_pickle=True) for f in files['test']]

        vocab = Lang(INIT_VOCAB)
        tra_data = encode(args, vocab, tra_files)
        val_data = encode(args, vocab, val_files)
        tst_data = encode(args, vocab, tst_files)

        with open(saved_file, "wb") as f:
            pickle.dump([tra_data, val_data, tst_data, vocab], f)
            logger.info("Saving data to {}.".format(saved_file))

    for i in range(2):
        logger.info("---------------- Example {} in Train Set ----------------".format(i))
        logger.info("[context]: {}".format([" ".join(utt) for utt in tra_data["context"][i]]))
        logger.info("[target]: {}".format([" ".join(tra_data["target"][i])]))
        logger.info("[emotion]: {}".format(["".join(tra_data["emotion"][i])]))
        logger.info("[situation]: {}".format([" ".join(tra_data["situation"][i])]))
        for j, cs in enumerate(tra_data["utt_cs"][i]):
            logger.info("[{}]: {}".format(RELATIONS[j], " ".join(_flatten(cs))))

    logger.info("Train Set Size: {}".format(len(tra_data['situation'])))
    logger.info("Valid Set Size: {}".format(len(val_data['situation'])))
    logger.info("Test Set Size: {}".format(len(tst_data['situation'])))
    return tra_data, val_data, tst_data, vocab


def prepare_emac_data_seq(args, vocab):
    saved_file = os.path.join('data/EMAC', "dataset_preproc.p")
    if os.path.exists(saved_file):
        logger.info("Loading data from {}.".format(saved_file))
        with open(saved_file, "rb") as f:
            [emac_tst_data, vocab] = pickle.load(f)
    else:
        logger.info("Building dataset from EQ-EMAC dataset.")
        emac_file_paths = ['data/EMAC/sys_{}_texts.test.npy'.format(x) for x in ['dialog', 'target', 'emotion', 'keyword']]
        emac_files = [np.load(f, allow_pickle=True) for f in emac_file_paths]
        emac_tst_data = encode(args, vocab, emac_files)

        with open(saved_file, "wb") as f:
            pickle.dump([emac_tst_data, vocab], f)
            logger.info("Saving data to {}.".format(saved_file))

        for i in range(2):
            logger.info("---------------- Example {} in EQ-EMAC Set ----------------".format(i))
            logger.info("[context]: {}".format([" ".join(utt) for utt in emac_tst_data["context"][i]]))
            logger.info("[target]: {}".format([" ".join(emac_tst_data["target"][i])]))
            logger.info("[emotion]: {}".format(["".join(emac_tst_data["emotion"][i])]))
            logger.info("[situation]: {}".format([" ".join(emac_tst_data["situation"][i])]))
            for j, cs in enumerate(emac_tst_data["utt_cs"][i]):
                logger.info("[{}]: {}".format(RELATIONS[j], " ".join(_flatten(cs))))

    emac_tst_set = Dataset(args, emac_tst_data, vocab)
    tst_loader = data.DataLoader(dataset=emac_tst_set, batch_size=1, shuffle=False, collate_fn=emac_tst_set.collate_fn)

    return tst_loader, vocab, len(emac_tst_set.emo_map)


def encode(args, vocab, files):
    data_dict = {"context": [], "target": [], "emotion": [], "situation": [], # "pol_cs": [],
                 "utt_cs": []}
    comet = Comet("data/Comet", args.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_context(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            for item in tqdm(items):
                item = tokenize(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break

    assert (len(data_dict['context']) == len(data_dict['target']) == len(data_dict['emotion']) == len(
        data_dict['situation']) == len(data_dict['utt_cs'])) # == len(data_dict['pol_cs'])

    return data_dict


def encode_context(vocab, items, data_dict, comet):
    for conv in tqdm(items):
        ctx_list = []
        for i, utt in enumerate(conv):
            item = tokenize(utt)
            ctx_list.append(item)
            vocab.index_words(item)
            if i == len(conv) - 1:
                cs_list = get_knowledge(comet, ctx_list, item)
                data_dict['utt_cs'].append(cs_list)

        data_dict['context'].append(ctx_list)


def get_knowledge(comet, context, item):
    cs_list = []
    input_event = " ".join(item)
    for r in RELATIONS:
        rel_cs = comet.generate(input_event, r)
        rel_cs = [tokenize(i) for i in rel_cs]
        cs_list.append(rel_cs)
    return cs_list


def get_polarity(analyzer, items):
    pol_list = []
    for item in items:
        pol = analyzer.polarity_scores(' '.join(_flatten(item)))
        pol_list.append(pol)
    return pol_list


def tokenize(sentence):
    sentence = sentence.lower()
    for k, v in WORD_PAIRS.items():
        sentence = sentence.replace(k, v)
    return nltk.word_tokenize(sentence)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x
