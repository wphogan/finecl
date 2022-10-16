import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import sklearn.metrics
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from src.transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp

# Local imports
from src.stage_3.sentence_level.dataset import REDataset
from src.stage_3.sentence_level.model import REModel


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(os.path.join(os.getcwd(), 'run.log')), 'a+') as f_log:
            f_log.write(s + '\n')


def f1_score(output, label, rel_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]
        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)
    return micro_f1, f1_by_relation


def set_seed(config):
    random.seed(config.trainer.seed)
    np.random.seed(config.trainer.seed)
    torch.manual_seed(config.trainer.seed)
    torch.cuda.manual_seed_all(config.trainer.seed)


def train(config, model, train_dataloader, dev_dataloader, test_dataloader, dual_run=False):
    # total step
    step_tot = len(train_dataloader) * config.trainer.max_epoch

    # optimizer
    if config.trainer.optim == "adamw":
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': config.trainer.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.trainer.lr, eps=config.trainer.adam_epsilon,
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.trainer.warmup_steps,
                                                    num_training_steps=step_tot)
    elif config.trainer.optim == "sgd":
        params = model.parameters()
        optimizer = optim.SGD(params, config.trainer.lr)
    elif config.trainer.optim == "adam":
        params = model.parameters()
        optimizer = optim.Adam(params, config.trainer.lr)

    # amp training
    if config.trainer.optim == "adamw":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Data parallel
    model = nn.DataParallel(model)
    model.train()
    model.zero_grad()

    logging("Begin train...")
    logging("We will train model in %d steps" % step_tot)
    global_step = 0
    best_dev_score = 0
    best_test_score = 0
    for i in range(config.trainer.max_epoch):
        for batch in train_dataloader:
            inputs = {
                "input_ids": batch[0],
                "mask": batch[1],
                "h_pos": batch[2],
                "t_pos": batch[3],
                "h_pos_l": batch[6],
                "t_pos_l": batch[7],
                "label": batch[4]
            }
            model.training = True
            model.train()
            loss, output = model(**inputs)
            if config.trainer.optim == "adamw":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.trainer.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            if config.trainer.optim == "adamw":
                scheduler.step()
            model.zero_grad()
            global_step += 1

            output = output.cpu().detach().numpy()
            label = batch[4].numpy()
            crr = (output == label).sum()
            tot = label.shape[0]

            sys.stdout.write("epoch: %d, loss: %.6f, acc: %.3f\r" % (i, loss, crr / tot))
            sys.stdout.flush()

        # dev
        with torch.no_grad():
            logging("")
            logging("deving....")
            model.training = False
            model.eval()

            if "semeval" in config.trainer.dataset or "tacred" in config.trainer.dataset:
                eval_func = eval_F1
            elif config.trainer.dataset == "wiki80" or config.trainer.dataset == "chemprot":
                eval_func = eval_ACC

            score = eval_func(config, model, dev_dataloader)
            if score > best_dev_score:
                best_dev_score = score
                best_test_score = eval_func(config, model, test_dataloader)
                logging("Best Dev score: %.3f,\tTest score: %.3f" % (best_dev_score, best_test_score))
            else:
                logging("Dev score: %.3f" % score)
            logging("-----------------------------------------------------------")

    logging("@RESULT: " + config.trainer.dataset + " Test score is %.3f" % best_test_score)

    # File name settings:
    model_name = config.model_name_or_path
    fname_out = f're_{config.trainer.dataset}_{model_name}_{str(config.trainer.train_prop)}_{config.trainer.seed}_{best_test_score:.3f}.log'

    f = open(os.path.join(os.getcwd(), fname_out), 'a+')
    if config.pretrained_model_path == "None":
        f.write("bert-base\t" + config.trainer.dataset + "\t" + str(
            config.trainer.train_prop) + "\t" + config.trainer.mode + "\t" + "seed:" + str(
            config.trainer.seed) + "\t" + "max_epoch:" + str(config.trainer.max_epoch) + "\t" + str(
            time.ctime()) + "\n")
    else:
        f.write(config.pretrained_model_path + "\t" + config.trainer.dataset + "\t" + str(
            config.trainer.train_prop) + "\t" + config.trainer.mode + "\t" + "seed:" + str(
            config.trainer.seed) + "\t" + "max_epoch:" + str(config.trainer.max_epoch) + "\t" + str(
            time.ctime()) + "\n")
    f.write("@RESULT: Best Dev score is %.3f, Test score is %.3f\n" % (best_dev_score, best_test_score))
    f.write("--------------------------------------------------------------\n")
    f.close()

    if dual_run:
        return best_test_score


def eval_F1(config, model, dataloader):
    tot_label = []
    tot_output = []
    for batch in dataloader:
        inputs = {
            "input_ids": batch[0],
            "mask": batch[1],
            "h_pos": batch[2],
            "t_pos": batch[3],
            "h_pos_l": batch[6],
            "t_pos_l": batch[7],
            "label": batch[4]
        }
        _, output = model(**inputs)
        tot_label.extend(batch[4].tolist())
        tot_output.extend(output.cpu().detach().tolist())

    f1, _ = f1_score(tot_output, tot_label, config.trainer.rel_num)
    return f1


def eval_ACC(config, model, dataloader):
    tot = 0.0
    crr = 0.0
    for batch in dataloader:
        inputs = {
            "input_ids": batch[0],
            "mask": batch[1],
            "h_pos": batch[2],
            "t_pos": batch[3],
            "h_pos_l": batch[6],
            "t_pos_l": batch[7],
            "label": batch[4]
        }
        _, output = model(**inputs)
        output = output.cpu().detach().numpy()
        label = batch[4].numpy()
        crr += (output == label).sum()
        tot += label.shape[0]

        sys.stdout.write("acc: %.3f\r" % (crr / tot))
        sys.stdout.flush()

    return crr / tot


def train_stage_3_sentence_re(config):
    logging('Experiment directory: ', os.getcwd())
    set_seed(config)

    dataset_dir = config.trainer.data_dir + config.trainer.dataset
    logging(f"Using {config.trainer.train_prop * 100}% train data from {dataset_dir}!")
    train_set = REDataset(dataset_dir, f"train_{config.trainer.train_prop}.json", config)
    dev_set = REDataset(dataset_dir, "dev.json", config)
    test_set = REDataset(dataset_dir, "test.json", config)

    logging('loading dataloader...')
    train_dataloader = data.DataLoader(train_set, batch_size=config.trainer.batch_size_per_gpu, shuffle=True)
    dev_dataloader = data.DataLoader(dev_set, batch_size=config.trainer.batch_size_per_gpu, shuffle=False)
    test_dataloader = data.DataLoader(test_set, batch_size=config.trainer.batch_size_per_gpu, shuffle=False)

    rel2id = json.load(open(os.path.join(dataset_dir, "rel2id.json")))
    config.trainer.rel_num = len(rel2id)

    logging('loading model...')
    model = REModel(config)
    model.cuda()
    train(config, model, train_dataloader, dev_dataloader, test_dataloader)
    logging('Results directory: ', os.getcwd())

    return
