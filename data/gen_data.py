import nltk
from nltk import word_tokenize
from itertools import chain, groupby
import numpy as np
import os

source_data = 'ag_news_csv'
dest_data = 'agnews'
dict_dir = '/ssd/important/155/workspace_wangshy/nlp_data/glove/glove.840B.300d.txt'

def clearText():
    print 'Cleaning Data...'
    data = []
    label_fid = open(dest_data + '/train_label.txt', 'w')
    #remove  useless symbols
    with open(source_data + '/train.csv') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\",\"')
            line[1] = line[1].replace('\\',' ').replace('\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            line[2] = line[2].replace('\\',' ').replace('\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            # subtract the offset of labels
            label_fid.write(str(int(line[0].split('\"')[1])-1)+'\n')
            data.append(line[1]+' '+line[2])
    train_num = len(data)

    label_fid.close()
    label_fid = open(dest_data+'/test_label.txt', 'w')
    with open(source_data+'/test.csv') as f:
        for line in f:
            line = line.strip('\n')
            line = str(line)
            line = line.split('\",\"')
            line[1] = line[1].replace('\\',' ').replace('\"\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            line[2] = line[2].replace('\\',' ').replace('\"\"',' ').replace('/',' ').replace('\'',' ').replace('*',' ').replace('-',' ').replace('.',' ').replace('(', ' ').replace(')', ' ')
            # subtract the offset of labels
            label_fid.write(str(int(line[0].split('\"')[1])-1)+'\n')
            data.append(line[1]+' '+line[2].rstrip('\"'))
    label_fid.close()
    return data, train_num

def gen_global_dict():
    print 'Generating global dict... It may take a few of seconds.'
    global_dict = {}
    with open(dict_dir) as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            global_dict[line[0]] = line[1]
    return global_dict

def gen_local_dict(global_dict, data):
    print 'Generating local dict...'
    separate_str = ' '
    tmp = separate_str.join(data)
    wordtoken = nltk.tokenize.word_tokenize(tmp.lower())
    words = list(np.unique(wordtoken))
    print str(len(words))+' different words in total.'

    local_dict = {}
    local_dict_fid = open(dest_data+'/local_dict.txt', 'w')
    local_fail_fid = open(dest_data+'/fail.txt', 'w')
    index = 0
    for i in range(0, len(words)):
        if global_dict.has_key(words[i]):
            local_dict[words[i]] = index
            index = index + 1
            local_dict_fid.write(words[i]+'\t'+global_dict[words[i]]+'\n')
        else:
            local_fail_fid.write(words[i]+'\n')
    print 'Local dict has been saved in \'local_dict.txt\'.'
    print 'Words that cannot be found in dict are saved in the \'fail.txt\''
    local_dict_fid.close()
    local_fail_fid.close()
    return local_dict

def gen_train_test(local_dict, data, train_num):
    train_fid = open(dest_data + '/train_data.txt', 'w')
    test_fid = open(dest_data + '/test_data.txt', 'w')
    print 'Begin to generate train/test data.'
    for i in range(len(data)):
        if i % 10000 == 0:
            print str(i) + ' samples have been generated!'
        words = nltk.tokenize.word_tokenize(data[i].lower())
        if i < train_num:
            for j in range(len(words)):
                if local_dict.has_key(words[j]):
                    train_fid.write(str(local_dict[words[j]])+' ')
            train_fid.write('\n')
        else:
            for j in range(len(words)):
                if local_dict.has_key(words[j]):
                    test_fid.write(str(local_dict[words[j]])+' ')
            test_fid.write('\n')
    train_fid.close()
    test_fid.close()

if __name__ == '__main__':
    global_dict = gen_global_dict()
    if (not os.path.exists(dest_data)):
        os.system('mkdir '+dest_data)
    data, train_num = clearText()
    print 'There are '+str(train_num)+' training samples and '+str(len(data) - train_num)+' testing samples.'
    local_dict = gen_local_dict(global_dict, data)
    gen_train_test(local_dict, data, train_num)
    print 'Finished!'




