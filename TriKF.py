import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.autograd import Variable
import numpy as np
import math
from tqdm import tqdm
import time
import random
from copy import deepcopy
from common_layer import *
from sklearn.metrics import accuracy_score
from model import Encoder, Decoder, Generator
from constants import *
from typing import Optional, Tuple


class TriKF(nn.Module):
    def __init__(self, args, vocab, decoder_number, model_file_path=None, load_optim=False):
        super(TriKF, self).__init__()
        self.args = args
        self.device = args.device
        self.is_eval = args.do_test
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.relations = RELATIONS

        self.embedding = share_embedding(args, self.vocab, args.pretrain_emb)

        self.encoder = self.generate_encoder(args.emb_dim)
        self.evt_encoder = self.generate_encoder(args.emb_dim)
        self.emo_encoder = self.generate_encoder(args.emb_dim)
        self.cog_encoder = self.generate_encoder(args.emb_dim)
        self.evt_ref_encoder = self.generate_encoder(2 * args.emb_dim)
        self.emo_ref_encoder = self.generate_encoder(2 * args.emb_dim)
        self.cog_ref_encoder = self.generate_encoder(2 * args.emb_dim)
        self.cross_evt = MultiHeadAttention(args.hidden_dim, args.hidden_dim, args.hidden_dim, args.hidden_dim,
                                            args.heads)
        self.cross_cog = MultiHeadAttention(args.hidden_dim, args.hidden_dim, args.hidden_dim, args.hidden_dim, args.heads)
        self.cross_emo = MultiHeadAttention(args.hidden_dim, args.hidden_dim, args.hidden_dim, args.hidden_dim, args.heads)
        self.fusion = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

        self.emo_lin = nn.Linear(args.hidden_dim, decoder_number, bias=False)
        self.pol_lin = nn.Linear(args.hidden_dim, 2, bias=False)
        self.klg_lin = MLP(args)

        self.decoder = Decoder(args, args.emb_dim, hidden_size=args.hidden_dim, num_layers=args.hop,
                               num_heads=args.heads, total_key_depth=args.depth, total_value_depth=args.depth,
                               filter_size=args.filter)
        self.generator = Generator(args, args.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)
        if args.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion_ppl = nn.NLLLoss(ignore_index=MAP_VOCAB['PAD'])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        if args.noam:
            self.optimizer = NoamOpt(
                args.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            logging.info("Loading weights.")
            state = torch.load(model_file_path, map_location=args.device)
            self.load_state_dict(state['model'])
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = args.output_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def generate_encoder(self, emb_dim):
        args = self.args
        return Encoder(args, emb_dim, args.hidden_dim, num_layers=args.hop, num_heads=args.heads,
                       total_key_depth=args.depth, total_value_depth=args.depth, filter_size=args.filter,
                       universal=args.universal)

    def save_model(self, running_avg_ppl, iteration):
        state = {
            'iter': iteration,
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl,
            'model': self.state_dict(),
        }

        model_save_path = os.path.join(self.model_dir, 'CEM_{}_{:.4f}'.format(iteration, running_avg_ppl))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def forward(self, batch): 
        enc_batch = batch['input_batch']  
        src_mask = enc_batch.data.eq(MAP_VOCAB['PAD']).unsqueeze(1)  
        mask_emb = self.embedding(batch['input_masks'])  
        src_emb = self.embedding(enc_batch) + mask_emb  
        enc_outputs = self.encoder(src_emb, src_mask) 

        # commonsense relations
        cs_embs = []  
        cs_masks = []  
        cs_outputs = []  
        for r in self.relations:
            emb = self.embedding(batch[r]).to(self.device)  
            mask = batch[r].data.eq(MAP_VOCAB['PAD']).unsqueeze(1) 
            cs_embs.append(emb)
            cs_masks.append(mask)
            if r in ["xIntent", "xNeed", "xWant", "xEffect"]:
                enc_output = self.cog_encoder(emb, mask)  
            elif r in ["Causes", "xReason"]:
                enc_output = self.evt_encoder(emb, mask)
            else:
                enc_output = self.emo_encoder(emb, mask)
            cs_outputs.append(enc_output)

        cls_tokens = [c[:, 0].unsqueeze(1) for c in cs_outputs]  
        evt_cls = cls_tokens[:2]
        cog_cls = cls_tokens[2:-1]
        emo_cls = torch.mean(cs_outputs[-1], dim=1).unsqueeze(1)  

        dim = [-1, enc_outputs.shape[1], -1]  
        emo_concat = torch.cat([enc_outputs, emo_cls.expand(dim)], dim=-1)  
        emo_outputs = self.emo_ref_encoder(emo_concat, src_mask)  

        emo_outputs, _ = self.cross_emo(enc_outputs, emo_outputs, emo_outputs, src_mask)
        emo_outputs = F.dropout(emo_outputs, p=0.1, training=self.training)
        gate = nn.Sigmoid()(self.fusion(torch.cat([enc_outputs, emo_outputs], dim=-1)))
        emo_ref_ctx = gate * enc_outputs + (1 - gate) * emo_outputs
        emo_logits = self.emo_lin(emo_ref_ctx[:, 0])
        pol_logits = self.pol_lin(emo_ref_ctx[:, 0])

        cog_outputs_list = []  
        for cls in cog_cls: 
            cog_concat = torch.cat([enc_outputs, cls.expand(dim)], dim=-1)  
            cog_concat_enc = self.cog_ref_encoder(cog_concat, src_mask) 
            cog_outputs_list.append(cog_concat_enc)
        cog_ref_ctx = []
        for cog_outputs in cog_outputs_list:
            cog_outputs, _ = self.cross_cog(enc_outputs, cog_outputs, cog_outputs, src_mask)
            cog_outputs = F.dropout(cog_outputs, p=0.1, training=self.training)
            gate = nn.Sigmoid()(self.fusion(torch.cat([enc_outputs, cog_outputs], dim=-1)))
            cog_outputs = gate * enc_outputs + (1 - gate) * cog_outputs
            cog_ref_ctx.append(cog_outputs)

        evt_outputs_list = []  
        for cls in evt_cls: 
            evt_concat = torch.cat([enc_outputs, cls.expand(dim)], dim=-1)  
            evt_concat_enc = self.evt_ref_encoder(evt_concat, src_mask) 
            evt_outputs_list.append(evt_concat_enc)
        evt_ref_ctx = []
        for evt_outputs in evt_outputs_list:
            evt_outputs, _ = self.cross_evt(enc_outputs, evt_outputs, evt_outputs, src_mask)
            evt_outputs = F.dropout(evt_outputs, p=0.1, training=self.training)
            gate = nn.Sigmoid()(self.fusion(torch.cat([enc_outputs, evt_outputs], dim=-1)))
            evt_outputs = gate * enc_outputs + (1 - gate) * evt_outputs
            evt_ref_ctx.append(evt_outputs)

        klg_ref_ctx = torch.cat(evt_ref_ctx + cog_ref_ctx + [emo_ref_ctx], dim=-1)
        klg_contrib = nn.Sigmoid()(klg_ref_ctx)
        klg_ref_ctx = klg_contrib * klg_ref_ctx
        klg_ref_ctx = self.klg_lin(klg_ref_ctx)

        return src_mask, klg_ref_ctx, emo_logits, pol_logits

    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(self.args, batch)
        dec_batch, _, _, _, _ = get_output_from_batch(self.args, batch)

        if self.args.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        src_mask, ctx_output, emo_logits, pol_logits = self.forward(batch)  # (16,1,60), (16,60,300), (16,32) (16,2)

        # Decode
        sos_token = torch.LongTensor([MAP_VOCAB['SOS']] * enc_batch.size(0)).unsqueeze(1).to(self.device)  # (16,1) 3333
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)  # (16,1) + (16,17) = (16,18)
        trg_mask = dec_batch_shift.data.eq(MAP_VOCAB['PAD']).unsqueeze(1)  # (16,1,18) False False ... True

        dec_emb = self.embedding(dec_batch_shift)  
        pre_logit, attn_dist = self.decoder(dec_emb, ctx_output, (src_mask, trg_mask))  # (16,18,300),(16,18,60)

        # compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if self.args.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        ) 

        emo_label = torch.LongTensor(batch["emotion_label"]).to(self.device)
        emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_label).to(self.device)
        pol_label = torch.LongTensor(batch['polarity_label']).to(self.device)
        pol_loss = nn.CrossEntropyLoss()(pol_logits, pol_label).to(self.device)
        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        
        loss = emo_loss + pol_loss + ctx_loss

        pred_emotion = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        emo_acc = accuracy_score(batch["emotion_label"], pred_emotion)
        pred_polarity = np.argmax(pol_logits.detach().cpu().numpy(), axis=1)
        pol_acc = accuracy_score(batch['polarity_label'], pred_polarity)

        # print results for testing
        top_preds = ""
        comet_res = {}

        if self.is_eval:
            top_preds = emo_logits.detach().cpu().numpy().argsort()[0][-3:][::-1]
            top_preds = f"{', '.join([MAP_EMO[pred.item()] for pred in top_preds])}"
            for r in self.relations:
                txt = [[" ".join(t) for t in tm] for tm in batch[f"{r}_text"]][0]
                comet_res[r] = txt

        if train:
            loss.backward()
            self.optimizer.step()

        return (
            ctx_loss.item(),
            math.exp(min(ctx_loss.item(), 100)),
            emo_loss.item(),
            emo_acc,
            top_preds,
            comet_res,
            pol_acc
        )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = self.args.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            _,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(self.args, batch)
        src_mask, ctx_output, _, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(MAP_VOCAB['SOS']).long().to(self.device)
        trg_mask = ys.data.eq(MAP_VOCAB['PAD']).unsqueeze(1)
        decoded_words = []
        for _ in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if self.args.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, trg_mask),
                )
            else:
                out, attn_dist = self.decoder(ys_embed, ctx_output, (src_mask, trg_mask))

            prob = self.generator(out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == MAP_VOCAB['EOS']
                    else self.vocab.idx2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(self.device)],
                dim=1,
            ).to(self.device)
            trg_mask = ys.data.eq(MAP_VOCAB['PAD']).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(self.args, batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(MAP_VOCAB['SOS']).long().to(self.device)
        trg_mask = ys.data.eq(MAP_VOCAB['PAD']).unsqueeze(1)
        decoded_words = []
        for _ in range(max_dec_step + 1):
            if self.args.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, trg_mask),
                )
            else:
                out, attn_dist = self.decoder(self.embedding(ys), ctx_output, (src_mask, trg_mask))

            logit = self.generator(out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == MAP_VOCAB['EOS']
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(self.device)],
                dim=1,
            ).to(self.device)
            trg_mask = ys.data.eq(MAP_VOCAB['PAD']).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        input_num = len(RELATIONS)
        input_dim = input_num * args.hidden_dim
        hid_num = input_num - 2 if input_num - 2 > 1 else input_num
        hid_dim = hid_num * args.hidden_dim
        out_dim = args.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


def get_input_from_batch(args, batch):
    enc_batch = batch['input_batch']
    enc_lens = batch['input_lengths']
    enc_text = batch['input_text']
    batch_size, max_enc_len = enc_batch.size()
    assert len(enc_lens) == batch_size

    enc_padding_mask = sequence_mask(args, enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    # args.pointer_gen

    c_t_1 = torch.zeros((batch_size, 2 * args.hidden_dim))

    # coverage
    coverage = None

    enc_padding_mask.to(args.device)
    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab.to(args.device)
    if extra_zeros is not None:
        extra_zeros.to(args.device)
    c_t_1.to(args.device)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, enc_text


def get_output_from_batch(args, batch):
    dec_batch = batch['target_batch']

    if args.pointer_gen:
        target_batch = batch['target_ext_vocab_batch']
    else:
        target_batch = dec_batch

    dec_lens = batch['target_lengths']
    max_dec_lens = max(dec_lens)

    assert max_dec_lens == target_batch.size(1)

    dec_padding_mask = sequence_mask(args, dec_lens, max_len=max_dec_lens).float()

    return dec_batch, dec_padding_mask, max_dec_lens, dec_lens, target_batch

