import csv

import numpy as np
from tkinter import _flatten
from nltk import word_tokenize


def preprocess(data_path):
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        cur_conv_id = ''
        conv = []

        files = {'train': [[], [], [], []], 'valid': [[], [], [], []], 'test': [[], [], [], []]}

        for line in reader:
            conv_id, utt_id, context, prompt, _, utt, set_id, _, q_num, _, act, intent, _, _ = line
            if cur_conv_id != conv_id:
                cur_conv_id = conv_id
                conv = []
            # can be a target: sys response and question count >= 1
            if int(utt_id) % 2 == 0 and int(q_num) > 0:
                dialog = conv.copy()
                target = utt.replace("_comma_", ",")
                emotion = context.replace("_comma_", ",")
                situation = prompt.replace("_comma_", ",")
                conv.append(utt)
                if set_id not in files.keys():
                    print("Error set_id: {}".format(set_id))
                else:
                    files[set_id][0].append(dialog)
                    files[set_id][1].append(target)
                    files[set_id][2].append(emotion)
                    files[set_id][3].append(situation)
            else:
                conv.append(utt.replace("_comma_", ","))

        for k, v in files.items():
            dialog_texts = np.empty((len(v[0]), ), dtype=object)
            for i in range(len(v[0])):
                dialog_texts[i] = v[0][i]
            target_texts = v[1]
            emotion_texts = v[2]
            situation_texts = v[3]
            # NOTE 备注CEM 的数据格式
            # sys_dialog_texts 是历史上下文
            # sys_target_texts 是当前应该生成的回复
            # sys_emotion_texts 是整个对话的情绪分类
            # sys_situation_texts 是prompt，可以算是对话的概要
            np.save('sys_dialog_texts.{}.npy'.format(k), dialog_texts)
            np.save('sys_target_texts.{}.npy'.format(k), target_texts)
            np.save('sys_emotion_texts.{}.npy'.format(k), emotion_texts)
            np.save('sys_situation_texts.{}.npy'.format(k), situation_texts)


def trans_act(s):
    s = s[1:-1]
    li = s.split(',')
    li = [x.strip(" ").strip("'") for x in li]
    return li


def analyze(data_path):
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        cur_conv_id = ''
        conv = []

        files = {'train': [[], [], [], [], []], 'valid': [[], [], [], [], []], 'test': [[], [], [], [], []]}

        for line in reader:
            conv_id, utt_id, context, prompt, _, utt, set_id, _, q_num, _, act, intent, _, _ = line
            if cur_conv_id != conv_id:
                cur_conv_id = conv_id
                conv = []
            # can be a target: sys response and question count >= 1
            if int(utt_id) % 2 == 0 and int(q_num) > 0:
                dialog = conv.copy()
                target = utt.replace("_comma_", ",")
                emotion = context.replace("_comma_", ",")
                situation = prompt.replace("_comma_", ",")
                act_list = trans_act(act)
                conv.append(utt)
                if set_id not in files.keys():
                    print("Error set_id: {}".format(set_id))
                else:
                    files[set_id][0].append(dialog)
                    files[set_id][1].append(target)
                    files[set_id][2].append(emotion)
                    files[set_id][3].append(situation)
                    files[set_id][4].append(act_list)
            else:
                conv.append(utt.replace("_comma_", ","))

        for k, v in files.items():
            dialog_texts = v[0]
            target_texts = v[1]
            emotion_texts = v[2]
            situation_texts = v[3]
            act_list = list(_flatten(v[4]))
            utt_num = 0
            utt_len = 0
            tg_len = 0
            emotions = {}
            acts = {}
            emotion_num = 0
            act_num = 0
            for conv in dialog_texts:
                utt_num += len(conv)
                for utt in conv:
                    utt_len += len(word_tokenize(utt))
            for tg in target_texts:
                tg_len += len(word_tokenize(tg))
            for emo in emotion_texts:
                if emo not in emotions.keys():
                    emotions[emo] = 1
                else:
                    emotions[emo] += 1
                emotion_num += 1
            for al in act_list:
                if al not in acts.keys():
                    acts[al] = 1
                else:
                    acts[al] += 1
                act_num += 1
            emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            acts = sorted(acts.items(), key=lambda  x: x[1], reverse=True)
            per_emotion = []
            per_act = []
            for key, value in emotions:
                per = round(100 * value / emotion_num, 2)
                per_emotion.append((key, per))
            for key, value in acts:
                per = round(100 * value / act_num, 2)
                per_act.append((key, per))

            print("------------ {} dataset ------------".format(k))
            print('Conversation Number in {} set: {}'.format(k, len(dialog_texts)))
            print('Utterance Number in {} set: {}/{}'.format(k, utt_num, (utt_num + len(target_texts))))
            print('Average Conversation Length in {} Set: {:.2f}/{:.2f}'.format(k, utt_num / len(dialog_texts),
                                                                        utt_num / len(dialog_texts) + 1))
            print('Average Utterance Length in {} set: {:.2f}/{:.2f}'.format(k, utt_len / utt_num,
                                                                     (utt_len + tg_len) / (utt_num + len(target_texts))))
            print('Average Target Utterance Length in {} Set: {}'.format(k, tg_len / len(target_texts)))
            print('Emotion Distribution in {} set: {}'.format(k, per_emotion))
            print('Act Distribution in {} set: {}'.format(k, per_act))


def analyzeED(data_dir):
    for k in ['train', 'dev', 'test']:
        dialog_texts = np.load(data_dir+"/sys_dialog_texts.{}.npy".format(k), allow_pickle=True)
        emotion_texts = np.load(data_dir + "/sys_emotion_texts.{}.npy".format(k), allow_pickle=True)
        target_texts = np.load(data_dir + "/sys_target_texts.{}.npy".format(k), allow_pickle=True)

        utt_num = 0
        utt_len = 0
        tg_len = 0
        emotions = {}
        emotion_num = 0
        for conv in dialog_texts:
            utt_num += len(conv)
            for utt in conv:
                utt_len += len(word_tokenize(utt))
        for tg in target_texts:
            tg_len += len(word_tokenize(tg))
        for emo in emotion_texts:
            if emo not in emotions.keys():
                emotions[emo] = 1
            else:
                emotions[emo] += 1
            emotion_num += 1
        emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        per_emotion = []
        for key, value in emotions:
            per = round(100 * value / emotion_num, 2)
            per_emotion.append((key, per))

        print("------------ {} dataset ------------".format(k))
        print('Conversation Number in {} set: {}'.format(k, len(dialog_texts)))
        print('Utterance Number in {} set: {}/{}'.format(k, utt_num, (utt_num + len(target_texts))))
        print('Average Conversation Length in {} Set: {:.2f}/{:.2f}'.format(k, utt_num / len(dialog_texts),
                                                                    utt_num / len(dialog_texts) + 1))
        print('Average Utterance Length in {} set: {:.2f}/{:.2f}'.format(k, utt_len / utt_num,
                                                                 (utt_len + tg_len) / (utt_num + len(target_texts))))
        print('Average Target Utterance Length in {} Set: {}'.format(k, tg_len / len(target_texts)))
        print('Emotion Distribution in {} set: {}'.format(k, per_emotion))


if __name__ == "__main__":
    data_dir = "../ED"
    data_path = "./ed_annotated.csv"
    # analyze(data_path)
    analyzeED(data_dir)







