import os
import logging
import random
import torch
import argparse
import wandb
import numpy as np
from copy import deepcopy
from tqdm import tqdm, trange
from torch.nn.init import xavier_uniform_
from dataloader import prepare_data_seq, make_infinite
from TriKF import TriKF
from common import *


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_random_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_params(params):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('--------- Parameter Settings ---------')
    print('-' * 80)
    for key in params.__dict__:
        if params.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, params.__dict__[key]).center(80))
    print('=' * 80)


def load_params():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/EQT",
                        help='processed EmpatheticDialogue dataset')
    parser.add_argument("--output_dir", type=str, default="output/test", help='output directory')
    parser.add_argument("--tokenizer_dir", type=str, default="data/", help='path to tokenization file')
    parser.add_argument("--emb_file", type=str, default='data/glove.6B.300d.txt', help='path to glove embedding file')

    # training
    parser.add_argument("--do_train", default=True, action='store_true', help="whether to run training")
    parser.add_argument("--do_test", default=True, action='store_true', help="whether to test")
    parser.add_argument("--model", type=str, default="seq2seq", help='model name, [KEMP, wo_ECE, wo_EDD]')
    parser.add_argument("--use_cuda", type=bool, default=True, help='gpu is available or not')
    parser.add_argument('--device_id', dest='device_id', type=str, default="0", help='gpu device id')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--eps', type=float, default=1e-9, help='arg in NoamOpt')
    parser.add_argument('--epochs', type=int, default=10000, help='training iterations')
    parser.add_argument('--check_iter', type=int, default=2000, help='validation iterations')
    parser.add_argument("--noam", default=True, action="store_true", help='NoamOpt')
    parser.add_argument("--learning_rate", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2, help='dropout')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size')
    parser.add_argument("--plm", action="store_true", help='use pretraining model or not')
    parser.add_argument("--use_oov_emb", action="store_true", help='')
    parser.add_argument("--pretrain_emb", default=True, action="store_true", help='use pretrained embedding (glove) or not')
    parser.add_argument("--weight_sharing", action="store_true",
                        help='sharing params between input embedding and output proj')
    parser.add_argument("--label_smoothing", default=True, action="store_true", help='label smoothing loss')
    parser.add_argument("--universal", action="store_true", help='universal transformer')
    parser.add_argument("--act", action="store_true", help='arg in universal transformer, adaptive computation time')
    parser.add_argument("--act_loss_weight", type=float, default=0.001, help='arg in universal transformer')
    parser.add_argument("--specify_model", action="store_true", help='arg for resuming training')

    # testing
    parser.add_argument("--beam_search", action="store_true", help='beam decoding')
    parser.add_argument("--beam_size", type=int, default=5, help='beam size')
    parser.add_argument("--topk", type=int, default=5, help='topk sampling')
    parser.add_argument("--project", action="store_true")

    # transformer
    parser.add_argument("--hidden_dim", type=int, default=300, help='hidden size')
    parser.add_argument("--emb_dim", type=int, default=300, help='embedding dimension')
    parser.add_argument("--hop", type=int, default=1, help='number of transformer layers')
    parser.add_argument("--heads", type=int, default=2, help='number of attention heads')
    parser.add_argument("--depth", type=int, default=40,
                        help='size of last dimension of keys/values. Must be divisible by number of heads')
    parser.add_argument("--filter", type=int, default=50, help='hidden size of the middle layer in FFN.')
    parser.add_argument("--projection", action="store_true",
                        help='project the input of decoder from embedding dimension to hidden dimension')
    parser.add_argument("--concept_num", type=int, default=1,
                        help='the maximum number of external concepts injection for a word.')
    parser.add_argument("--total_concept_num", type=int, default=10,
                        help='the maximum number of external concepts injection for a sentence.')
    parser.add_argument("--max_seq_length", type=int, default=1000,
                        help='max sequence length (required for timing signal)')
    parser.add_argument("--pointer_gen", action="store_true", help='copy mechanism')
    parser.add_argument("--emotion_feature", action="store_true", help="emotional feature")

    args = parser.parse_args()
    print_params(args)
    args.collect_stats = False

    args.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    return args


def evaluate(model, data, eval_type='valid', max_dec_step=30, output_file=None):
    model.__id__logger = 0
    if eval_type == 'test':
        logger.info("--------- Start Testing ---------")
    else:
        logger.info("--------- Start Evaluating ---------")
    logger.info("\tExample Size: {}".format(len(data)))

    refs, hyp_g, results = [], [], []
    loss, ppl, bce, acc, pol_acc = [], [], [], [], []

    pbar = tqdm(enumerate(data), total=len(data))
    # translator = Translator(model, model.vocab)

    for _, batch in pbar:
        l, p, bce_prog, acc_prog, top_preds, comet_res, pol_acc_prog = model.train_one_batch(batch, 0, train=False)
        loss.append(l)
        ppl.append(p)
        bce.append(bce_prog)
        acc.append(acc_prog)
        pol_acc.append(pol_acc_prog)
        if eval_type == 'test':
            sent_g = model.decoder_greedy(batch, max_dec_step=max_dec_step)
            # sent_b = translator.beam_search(batch, max_dec_step=max_dec_step)
            for i, greedy_sent in enumerate(sent_g):
                rf = ' '.join(batch['target_text'][i])
                hyp_g.append(greedy_sent)
                # hyp_b = sent_b[i]
                refs.append(rf)
                tmp = print_custom(
                    emotion=batch['emotion_text'][i],
                    dial=[' '.join(s) for s in batch['input_text'][i]],
                    ref=rf,
                    hyp_b='',
                    hyp_g=greedy_sent,
                    pred_emotions=top_preds,
                    comet_res=comet_res,
                )
                results.append(tmp)

        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l))))

    loss = np.mean(loss)
    ppl = np.mean(ppl)
    bce = np.mean(bce)
    acc = np.mean(acc)
    pol_acc = np.mean(pol_acc)

    logger.info("--------- Evaluation Results ---------")
    logger.info("Loss: {:.4f}\tPPL: {:.4f}\tAccuracy: {:.4f} {:.4f}\n".format(loss, math.exp(loss), acc, pol_acc))

    return loss, math.exp(loss), bce, acc, results, pol_acc


def test(args, model, test_set):
    model.is_eval = True
    loss_test, ppl_test, bce_test, acc_test, results, pol_acc_test = evaluate(model, test_set, eval_type='test', max_dec_step=50)
    output_file = os.path.join(args.output_dir, 'results.txt')
    with open(output_file, 'w') as f:
        f.write("EVAL\tLoss: {:.4f}\tPPL: {:.4f}\tBCE: {:.4f}\tAccuracy: {:.4f} {:.4f}\n".format(loss_test, ppl_test, bce_test, acc_test, pol_acc_test))
        for r in results:
            f.write(r)


def main():
    args = load_params()
    set_random_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''.format(args.device_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device_id))

    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(args, batch_size=args.batch_size)

    model = TriKF(args=args, vocab=vocab, decoder_number=dec_num)
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and args.pretrain_emb):
            xavier_uniform_(p)
    logger.info("Trainable Parameters: {}".format(count_parameters(model)))

    if args.do_train:
        logger.info("--------- Start Training ---------")
        logger.info("\tExample Size: {}".format(len(train_set)))
        logger.info("\tBatch Size: {}".format(args.batch_size))
        loss_val, ppl_val, bce_val, acc_val, pol_acc_val = None, None, None, None, None
        weights_best = None

        try:
            model = model.to(args.device)
            model.train()
            best_ppl = 1000
            patient = 0
            weights_best = deepcopy(model.state_dict())
            data_iter = make_infinite(train_set)

            for n_iter in tqdm(range(1000000)):
                loss, ppl, bce, acc, _, _, pol_acc = model.train_one_batch(next(data_iter), n_iter)
                # wandb.log({'tr_loss': loss.item(), 'tr_ppl': ppl.item(), 'tr_bce': bce.item(), 'tr_acc': acc.item()})
                # if args.noam:
                #     wandb.log({'lr': model.optimizer._rate})

                # evaluating
                if (n_iter + 1) % args.check_iter == 0:
                    model.eval()
                    model.epoch = n_iter
                    loss_val, ppl_val, bce_val, acc_val, _, pol_acc_val = evaluate(model, dev_set, eval_type='valid',
                                                                      max_dec_step=50)
                    # wandb.log({'ppl_val': ppl_val, 'acc_val': acc_val})
                    model.train()

                    if n_iter < 9000:
                        continue
                    if ppl_val <= best_ppl:
                        best_ppl = ppl_val
                        patient = 0
                        torch.save({'models': model.state_dict(),
                                    'result': [loss_val, ppl_val, bce_val, acc_val, pol_acc_val]},
                                   os.path.join(args.output_dir, 'model_{}_{:.4f}.tar'.format(n_iter, best_ppl)))
                        weights_best = deepcopy(model.state_dict())
                        logger.info("Best PPL: {} \t Patient: {}".format(best_ppl, patient))
                    else:
                        patient += 1
                        logger.info("Current patient: {}".format(patient))
                    # early stop
                    if patient > 2:
                        logger.info("Early Stop.")
                        break

        except KeyboardInterrupt:
            logger.info("--------- Exiting from training early ---------")

        torch.save({'models': weights_best,
                    'result': [loss_val, ppl_val, bce_val, acc_val, pol_acc_val]},
                   os.path.join(args.output_dir, 'model_best.tar'))
        logger.info("Saving the best checkpoints to {}.".format(os.path.join(args.output_dir, 'model_best.tar')))

    if args.do_test:
        checkpoints = torch.load(os.path.join(args.output_dir, 'model_best.tar'))
        weights_best = checkpoints['models']
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        model = model.to(args.device)
        model.eval()
        model.epoch = 100
        with torch.no_grad():
            test(args, model, test_set)


if __name__ == '__main__':
    main()
