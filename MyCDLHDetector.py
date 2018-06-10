from __future__ import print_function

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
from tempfile import NamedTemporaryFile
import shutil

import argparse
import sys
import pickle

from myVisitor import MyVisitor

#from pycparser import c_parser, c_ast, parse_file

torch.manual_seed(1)

# def preprocess():
#     """
#     Preprocess datasets. Get all ASTs.

#     Return:
#         (training_data, test_data)
#     """
#     training_data = []
#     test_data = {}
#     training_dir = './train'
#     test_dir = './test'

#     for dirpath, dirnames, filenames in os.walk(training_dir):
#         if '.DS_Store' in filenames:
#             filenames.remove('.DS_Store')
#         for filename in filenames:
#             #print(os.path.join(dirpath, filename))
#             filepath = os.path.join(dirpath, filename)
#             try:
#                 ast = parse_file(filepath, use_cpp=False)
#                 cv = MyVisitor()
#                 cv.visit(ast)
#                 training_data.append(cv.values)
#             except Exception as e:
#                 print('error when parsing', filepath, "type:", type(e))
#                 training_data.append([])
#     #save training_data into file
#     with open('training_data.txt', 'wb') as f:
#         pickle.dump(training_data, f)
#     print("training_data length: ", len(training_data))
#     print("training_data[0]:", training_data[0])

#     for dirpath, dirnames, filenames in os.walk(test_dir):
#         if '.DS_Store' in filenames:
#             filenames.remove('.DS_Store')
#         for filename in filenames:
#             filepath = os.path.join(dirpath, filename)
#             try:
#                 ast = parse_file(filepath, use_cpp=False)
#                 cv = MyVisitor()
#                 cv.visit(ast)
#                 test_data[filename] = cv.values
#             except Exception as e:
#                 print('error when parsing', filepath, "type", type(e))
#     #save test_data into files
#     with open('test_data.txt', 'wb') as f:
#         pickle.dump(test_data, f)
#     print("test_data length: ", len(test_data))
#     print("test_data[0]:", test_data[0])

#     #save word_to_ix to file
#     word_to_ix = {}
#     for sentence in training_data:
#         for word in sentence:
#             if word not in word_to_ix:
#                 word_to_ix[word] = len(word_to_ix)
#     for sentence in test_data:
#         for word in sentence:
#             if word not in word_to_ix:
#                 word_to_ix[word] = len(word_to_ix)
#     with open('word_to_ix.txt', 'wb') as f:
#         pickle.dump(word_to_ix, f)

def get_datas_from_files():
    with open('training_data.txt', 'rb') as f:
        training_data = pickle.load(f)
    with open('test_data.txt', 'rb') as f:
        test_data = pickle.load(f)
    with open('word_to_ix.txt', 'rb') as f:
        word_to_ix = pickle.load(f)
    return training_data, test_data, word_to_ix

def prepare_sequence(sequence, to_inx):
    idxs = [to_inx[w] for w in sequence]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

EMBEDDING_DIM = 64
HIDDEN_DIM = 32

class MyModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(MyModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        ##LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)), 
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_scores = F.log_softmax(lstm_out.view(len(sentence), -1), dim=1)
        return tag_scores
def train_v1(training_data, test_data, word_to_ix):
    model = MyModule(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), HIDDEN_DIM)
    loss_funciton = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    num_types = len(training_data) // 500

    # generate training_label
    training_label = []
    for i in range(num_types):
        for j in range(500):
            training_label.append(i)
    #print(training_label[0], training_label[-1])

    #start to train
    i = 0
    for epoch in range(2):
        for (sentence, tag) in zip(training_data, training_label):
            if len(sentence)==0:
                continue
            model.zero_grad()
            model.hidden = model.init_hidden()

            #print("tag", tag)
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = autograd.Variable(torch.LongTensor([tag]))

            tag_scores = model(sentence_in)
            #print(tag_scores[-1],targets)
            loss = loss_funciton(tag_scores[-1].view(1, -1), targets)
            loss.backward()
            if i % 1999 == 0:
                print(i+1," loss: ", loss)
            optimizer.step()
            i+=1
    torch.save(model, "model_v1.pt")

    #用model给所有的testdata进行标记
    print('start to testdata')
    test_labels = {}
    count = 0
    for key, ast in test_data.items():
        sentence = prepare_sequence(ast, word_to_ix)
        target = model(sentence)[-1]
        prediction = torch.max(target, 0)[1].data[0]
        test_labels[key] = prediction
        count+=1
        if (count % 1000) == 1:
            print(count)
    #存入文件
    print('finish test_label')
    with open('test_labels_v1.txt', 'wb') as f:
        pickle.dump(test_labels, f)

def get_result_csv_v1(model, test_data, word_to_ix):
    fields = ['id1_id2', 'predictions']
    tempfile = NamedTemporaryFile(mode='w', delete=False)
    
    with open('test_labels_v1', 'rb') as f:
        test_labels = pickle.load(f)
    with open('sample_submission.csv', 'r') as fr, tempfile:
        reader = csv.DictReader(fr)
        writer = csv.DictWriter(tempfile, fieldnames=fields)
        for row in reader:
            id1, id2 = row['id1_id2'].split('_')
            id1_file = id1 + '.txt'
            id2_file = id2 + '.txt'
            # sentence_1 = prepare_sequence(test_data[id1_file], word_to_ix)  
            # sentenct_2 = prepare_sequence(test_data[id2_file], word_to_ix)

            #print(file_id1, file_id2)
            # target_id1 = model(sentence_1)[-1]
            # target_id2 = model(sentenct_2)[-1]
            #print(target_id1)
            prediction1 = test_labels[id1_file]
            prediction2 = test_labels[id2_file]
            #print(target_id1, target_id2)
            #print(prediction1, prediction2)
            if (prediction1 == prediction2):
                writer.writerow({'id1_id2': row['id1_id2'], 'predictions': 1})
            else:
                writer.writerow({'id1_id2': row['id1_id2'], 'predictions': 0})
            
    shutil.move(tempfile.name, 'my_submission.csv')

class MyModule2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(MyModule2, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        ##LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)), 
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_scores = F.log_softmax(lstm_out.view(len(sentence), -1), dim=1)
        return tag_scores
def train_v2(training_data, test_data, word_to_ix):
    model = MyModule2(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), HIDDEN_DIM)
    if (torch.cuda.is_available()):
        model = model.cuda()
    loss_funciton = nn.HingeEmbeddingLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    #start to train
    i = 0
    for epoch in range(1):
        #训练相同的代码对
        epoch_loss = 0
        running_loss = 0
        same_pairs = list(zip(training_data[::2], training_data[1::2]))
        for sentence_1, sentence_2 in same_pairs:
            if len(sentence_1)== 0 or len(sentence_2) == 0:
                continue
            model.zero_grad()
            model.hidden = model.init_hidden()

            #print("tag", tag)
            sentence_in_1 = prepare_sequence(sentence_1, word_to_ix)
            sentence_in_2 = prepare_sequence(sentence_2, word_to_ix)
            if (torch.cuda.is_available()):
                sentence_in_1, sentence_in_2 = sentence_in_1.cuda(),  sentence_in_2.cuda()

            tag_scores_1 = model(sentence_in_1)[-1]
            tag_scores_2 = model(sentence_in_2)[-1]
            #print(tag_scores_1, tag_scores_2)
            distance = F.pairwise_distance(tag_scores_1.view(1,-1), tag_scores_2.view(1,-1),p=1)
            #print(tag_scores[-1],targets)
            #print(distance.data, torch.FloatTensor([1.0]))
            loss = loss_funciton(distance, autograd.Variable(torch.FloatTensor([1.0])))
            loss.backward()
            optimizer.step()

            #print statistics
            running_loss += loss.data[0]
            epoch_loss += loss.data[0]
            i+=1
            if i % 200 == 199:
                print(epoch, i+1,"running loss: ", running_loss/200)
                running_loss = 0
    print('finish to train same codes')
    # 训练不同的代码对
    for epoch in range(1):
        epoch_loss = 0
        running_loss = 0
        for i in range(82):
            print(i*500)
            for j in range(500):
                if len(training_data[i*500+j]) == 0 or len(training_data[i*500+500+j]) == 0:
                    continue
                model.zero_grad()
                model.hidden = model.init_hidden()

                sentence_in_1 = prepare_sequence(training_data[i*500+j], word_to_ix)
                sentence_in_2 = prepare_sequence(training_data[i*500+500+j], word_to_ix)
                if (torch.cuda.is_available()):
                    sentence_in_1, sentence_in_2 = sentence_in_1.cuda(),  sentence_in_2.cuda()
                tag_scores_1 = model(sentence_in_1)[-1]
                tag_scores_2 = model(sentence_in_2)[-1]
                distance = F.pairwise_distance(tag_scores_1.view(1,-1), tag_scores_2.view(1,-1),p=1)
            
                loss = loss_funciton(distance, autograd.Variable(torch.FloatTensor([-1.0])))
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                #print(loss.data[0])
                epoch_loss += loss.data[0]
                if j % 100 == 0:
                    print(distance)
                    print(epoch, i+1,"running loss: ", running_loss/100)
                    running_loss = 0
    print('finish to train different codes')
    

    torch.save(model, "model_v2.pt")

def generate_result(model, test_data, word_to_ix):
    results = {}
    with open('sample_submission.csv', 'r') as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            results[row['id1_id2']] = row['predictions']
    #start to get my results
    for id1_id2, predictioins in results.items():
        id1, id2 = id1_id2.split('_')
        id1_file = id1 + '.txt'
        id2_file = id2 + '.txt'
        sentence_1 = test_data[id1_file]
        sentence_2 = test_data[id2_file]

        sentence_in_1 = prepare_sequence(sentence_1, word_to_ix)
        sentence_in_2 = prepare_sequence(sentence_2, word_to_ix)
        if (torch.cuda.is_available()):
            sentence_in_1, sentence_in_2 = sentence_in_1.cuda(),  sentence_in_2.cuda()

        tag_scores_1 = model(sentence_in_1)[-1]
        tag_scores_2 = model(sentence_in_2)[-1]
        #print(tag_scores_1, tag_scores_2)
        distance = F.pairwise_distance(tag_scores_1.view(1,-1), tag_scores_2.view(1,-1),p=1)
        print(distance)

    
if __name__ == '__main__':
    #preprocess()
    training_data, test_data, word_to_ix = get_datas_from_files()
    #model_v1 = torch.load('model_v1.pt')
    model = torch.load('model_v2.pt')
    #train_v1(training_data, test_data, word_to_ix)
    train_v2(training_data, test_data, word_to_ix)
    #generate_result(model_v2, test_data, word_to_ix)
    
