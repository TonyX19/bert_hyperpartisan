# from bert_serving.client import BertClient
# bc = BertClient()
base_dir = '/home/hechen/hyperpartisan/data/'

import time

import xml.etree.ElementTree as ET
import xml.dom.minidom as xmldom
import os
import spacy

train_xml = os.path.abspath(base_dir+"/ground-truth-training-bypublisher.xml")
test_xml = os.path.abspath(base_dir+"/ground-truth-test-byarticle.xml")
#####

tokenlization = spacy.load('en', disable=['parser', 'ner'])

def customTokenize(text):
    '''
    lower, strip numbers and punctuation, remove stop words
    '''
    tokens = tokenlization(text)
    words = [word.lemma_ for word in tokens if word.lemma_.isalpha()]

    return words


def extractLabelXml(xml_file):
    xml_tree = ET.parse(xml_file)
    tree_root = xml_tree.getroot()

    label_pos_dict = {}
    label_neg_dict = {}
    for child in tree_root:
        if child.attrib['hyperpartisan'] == 'true':
            label_pos_dict[child.attrib['id']] = 'true'
        else:
            label_neg_dict[child.attrib['id']] = 'false'

    return [label_pos_dict,label_neg_dict]

print("loading ground truth xml.....")   
train_label_pos_dict,train_label_neg_dict = extractLabelXml(train_xml)
test_label_pos_dict,test_label_neg_dict = extractLabelXml(test_xml)
print("ground truth xml loaded.....")   



train_txt = base_dir + '/articles-training-bypublisher.txt.cleaned'
test_txt = base_dir + '/articles-test-byarticle.txt.cleaned'

train_sentence_dict = {}
test_sentence_dict = {}

def loadText(text_dir):
    text_f = open(text_dir,'r')
    sentence_dict = {}
    for line in text_f:
        _id,sent = line.split("\t")
        #sent_rsw = remove_stopwords(lemmatization(sent.rstrip()))
        if _id in sentence_dict.keys():
            sentence_dict[_id].append(sent)
            continue;
        sentence_dict[_id] = [sent]   

    return sentence_dict;

s_time = time.time()
print('train_sentence_dict loading')
train_sentence_dict = loadText(train_txt);
print('train_sentence_dict loaded')
print(str(time.time()-s_time) + " cost")
from random import sample

y_pos_train = sample(list(train_label_pos_dict),10000)
y_neg_train = sample(list(train_label_neg_dict),10000)

y_pos_test = list(test_label_pos_dict)
y_neg_test = list(test_label_neg_dict)

x_pos_train_list = []
x_neg_train_list = []
print('data seperating')
i = 1
for _id in y_pos_train:
    print('pos ',str(_id))
    print('pos',str(i))
    i += 1
    sent_list = []
    for sent in train_sentence_dict[_id]:
        token_list = customTokenize(sent)
        sent_list.extend(token_list)
    x_pos_train_list.extend(sent_list)

i = 1
for _id in y_neg_train:
    print('neg ',str(_id))
    print('neg',str(i))
    i += 1
    sent_list = []
    for sent in train_sentence_dict[_id]:
        token_list = customTokenize(sent)
        sent_list.extend(token_list)
    x_pos_train_list.extend(sent_list)

##############test##########
import pickle;

print('dumping......')
train_pos_txt = base_dir + '/training_pos_sent.txt.new'
train_neg_txt = base_dir + '/training_neg_sent.txt.new'
f = open(train_pos_txt, 'wb')
pickle.dump(x_pos_train_list, f)

f = open(train_neg_txt, 'wb')
pickle.dump(x_neg_train_list, f)

