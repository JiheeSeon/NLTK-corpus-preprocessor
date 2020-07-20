#module import
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import *
from nltk.classify import MaxentClassifier
from urllib import request
from bs4 import BeautifulSoup
from bs4.element import Tag
from time import sleep
import re
import os


### Step 1. Crawling gni corpus with BeautifulSoup ###

# Download raw file for genominfo
def download(dirr, a, b):
    sp_flag = 0 if b - a == 1 else 1
    complete_flag = 0
    print("Start download...")
    for i in range(a, b):
        try:
            url = 'https://genominfo.org/journal/view.php?number='
            fnum = str(i)
            html = None
            with request.urlopen(url + fnum) as req:
                html = req.read()
                soup = BeautifulSoup(html, 'html.parser')

                decompose_arr = ['div.fig.panel', 'div.table-wrap.panel', 'title', 'div#article-back.back']
                for i in decompose_arr:
                    for tag in soup.select(i):
                        tag.decompose()

                raw = soup.find('div', {'id': 'include_journal'}).get_text()
                raw = re.sub(r'\xa0', '', raw)
                with open(dirr + "/raw%s.txt" % fnum, "w+", encoding='UTF-8') as f:
                    f.write(raw)
                    f.close()
                req.close()
                if complete_flag == 0:
                    complete_flag = 1
        except AttributeError:
            pass

    if not (complete_flag | sp_flag):
        print("해당 페이지가 존재하지 않습니다.")
    elif complete_flag == 1:
        print("다운로드를 완료하였습니다.")
    else:
        print("입력한 모든 페이지들이 존재하지 않습니다.")


print("=============== GNI Corpus Downloader ===============")
dirr = input("다운로드를 원하는 폴더를 선택해주세요.")
i = input("i. 전체 데이터 다운로드를 원하면 all\n"
          "ii. 업데이트를 원하면 update\n"
          "iii. 특정 페이지 다운로드를 원하면 page number 또는 url 주소\n"
          "iv. 특정 구간의 여러 페이지들을 다운로드하기 원하면 [시작 페이지번호,마지막 페이지번호]형태로 입력해주세요\n")

# 사용자가 원하는 페이지 내용을 raw 형태로 담을 raw 폴더 생성
try:
    dirr_raw = dirr + "/raw"
    if not os.path.isdir(dirr_raw):
        os.mkdir(dirr_raw)
except:
    print("File Creation Error")

# Input handling을 위한 정규식 변수
p = re.compile(r"(http|https)://genominfo\.org/journal/view\.php\?number=[\d]*")
num_p = re.compile("(^[0-9]*$)")
interval_p = re.compile(r"\[(?P<start>[\d]*),(?P<fin>[\d]*)\]")

# Input handling
if i == 'all':
    download(dirr_raw, 1, 999)

elif i == 'update':
    test = os.listdir(dirr_raw)
    test.sort(key=lambda f: int(re.sub('\D', '', f)))
    m = re.match(r"raw(?P<num>[\d]*)\.txt", test[-1])
    download(dirr_raw, int(m.groupdict().get('num')) + 1, 1000)

elif interval_p.match(i):
    m = re.match(interval_p, i)
    start = m.groupdict().get('start')
    fin = m.groupdict().get('fin')
    download(dirr_raw, int(start), int(fin) + 1)

elif p.match(i):
    ind = int(re.sub("https://genominfo.org/journal/view.php\?number=", "", i))
    download(dirr_raw, ind, ind + 1)

elif num_p.match(i):
    download(dirr_raw, int(i), int(i) + 1)

else:
    print("Wrong input type.")


### Step 2. Make Train set and Train various classifier ###

#train set directory and Reader
gni_root_train = ".\example\train_set" ##change into your train folder
gni_reader_train = nltk.corpus.PlaintextCorpusReader(gni_root_train, ".*\.txt", encoding="utf-8")

#Downloaded file directory and Reader
gni_root_raw = dirr_raw
gni_reader_raw = nltk.corpus.PlaintextCorpusReader(gni_root_raw, ".*\.txt", encoding="utf-8")

#Evaluate file directory and Reader include reference
gni_root_eval = ".\example\evaluate"
gni_reader_eval = nltk.corpus.PlaintextCorpusReader(gni_root_eval, ".*\.txt", encoding="utf-8")

#Evaluate file directory and Reader without reference
gni_root_eval_no = ".\example\evaluateDel"
gni_reader_eval_no = nltk.corpus.PlaintextCorpusReader(gni_root_eval_no, ".*\.txt", encoding="utf-8")

#segment sentences
split_sent = gni_reader_train.raw().split('\n')
prepro = [sent for sent in split_sent if sent != '']
sents = [nltk.tokenize.word_tokenize(sent) for sent in prepro]

#labeling sents and boundary
tokens = []
boundaries = set() #문장의 끝 부호의 인덱스들을 저장할 집합
offset = 0
for sent in sents:
    tokens.extend(sent) 
    offset += len(sent) # 문장의 길이 누적값
    boundaries.add(offset-1) # 문장의 끝 인덱스번호를 저장(즉, 그 인덱스 번호는 문장의 마침표이다)
    
def find_sent_finish(tokens, i): #feature extractor
    return {
        'prev_word' : tokens[i-1].lower(),
        'is_prev_word_sci_abbr': tokens[i-1] in ['et','al','etc','Dr','Prof','Acc','No','Fig','Vol'],
        'punct': tokens[i],
        'is_prev_word_one_char': len(tokens[i-1]) == 1,
        'is_next_word_first_char_upper': tokens[i+1][0].isupper()
    }

#make Featuresets
featuresets = [(find_sent_finish(tokens, i), (i in boundaries)) 
               for i in range(1, len(tokens)-1)
              if tokens[i] in '.?!']

#Divide train sets and test sets
size = int(len(featuresets) * 0.9)
train_set, test_set = featuresets[:size], featuresets[size:]

#Classifier 1 : NaiveBayes Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Classifier 2 : DecisionTree Classifier
decisionClassifier = nltk.classify.DecisionTreeClassifier.train(train_set)

#Classifier 3 : Max Entrophy Classifier
maxentClassifier = MaxentClassifier.train(train_set)


## Step 2.2 Make NaiveBayesClassifier with Brown corpus ###

#To compare, use brown corpus
brown_corpus = nltk.corpus.brown
prepro_brown = brown_corpus.sents()

#segement brown sentences
tokens_brown = []
boundaries_brown = set() #문장의 끝 부호의 인덱스들을 저장할 집합
offset_brown = 0
for sent in prepro_brown:
    tokens_brown.extend(sent)
    offset_brown += len(sent) # 문장의 길이 누적값
    boundaries_brown.add(offset_brown-1) # 문장의 끝 인덱스번호를 저장(즉, 그 인덱스 번호는 문장의 마침표이다)

#make brown featuresets
featuresets_brown = [(find_sent_finish(tokens_brown, i), (i in boundaries_brown))
               for i in range(1, len(tokens_brown)-1)
              if tokens_brown[i] in '.?!']

#make brown featuresets , train brown train sets and evaluate accuracy
size_brown = int(len(featuresets_brown) * 0.9)
train_set_brown, test_set_brown = featuresets_brown[:size_brown], featuresets_brown[size_brown:]
classifier_brown = nltk.NaiveBayesClassifier.train(train_set_brown)
nltk.classify.accuracy(classifier_brown,test_set_brown)

#show most informative features
classifier_brown.show_most_informative_features(10)



### Step 3. Do sentence segmentation with Model ###

#segment raw file
split_sent2 = gni_reader_raw.raw().split('\n')
prepro2 = [sent for sent in split_sent2]
sents2 = [nltk.tokenize.word_tokenize(sent) for sent in prepro2 if len(sent) > 1]

#labeling raw files
tokens2 = []
boundaries2 = set()
offset2 = 0
for sent in sents2:
    tokens2.extend(sent)
    offset2 += len(sent)
    boundaries2.add(offset2-1)

# make raw file featuresets
featuresets2 = [(find_sent_finish(tokens2, i), (i in boundaries2))
               for i in range(1, len(tokens2)-1)
              if tokens2[i] in '.?!']
# '.:)"\'”]|'



### Step 3.1 Save Segmented Files ###

#load segmented files
gni_root_segmented = dirr+"/segmented"
#function to segment_sentence
def segment_sentences(words):
    start = 0
    sents = []
    for i, word in enumerate(words):
        if i == len(words)-1: continue
        if word in '.?!' and classifier.classify(find_sent_finish(words, i)) == True:
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents

#function to concatnate each file
def re_classfy(file):
    prepro2_file = gni_reader_raw.raw(fileids=[file]).split('\n')
    sents2_file = [nltk.tokenize.word_tokenize(sent) for sent in prepro2_file if len(sent) > 1]
    resents2_file = [segment_sentences(sent) for sent in sents2_file]
    final_file = []

    for i in resents2_file:
        final_file.extend(i)

    fsent_file = []
    for j in final_file:
        sent = ''
        for word in j:
            if word == '': continue
            if word in '{}[]()\.,?!:/;<>\'' or word == j[0]:
                sent += word
            else: sent += ' '+ word
        fsent_file.append(sent)

    fsent_file_str = ''
    for i in fsent_file:
        fsent_file_str += i+'\n'
    return fsent_file_str

#function to do post process
def postpro(matchObj):
    s = str(matchObj.group())
    return ({"( ":"(",
             "[ ":"[",
             "< ":"<",
             "{ ":"{",
             " %":"%",
             ": //":"://"}.get(s))


# Make directory
try:
    if not os.path.isdir(gni_root_segmented):
        os.mkdir(gni_root_segmented)
except:
    print("File Creation Error")

for i in range(1, len(gni_reader_raw.fileids()) + 1):
    try:
        after_str = re_classfy("raw" + str(i) + ".txt")
        final_str = re.sub(r"\( |\[ |\< |\{ |\ %|: //", postpro, after_str)
        f = open(gni_root_segmented + '/' + "seg" + str(i) + ".txt", 'w', encoding='utf-8')
        f.write(final_str)
        f.close()
    except:
        pass
print("Segmented file 생성을 완료하였습니다")


### Step 4. Make evaluation set and calculate accuracy ###

#segment evaluation files
split_sent_eval = gni_reader_eval.raw().split('\n')
prepro_eval = [sent for sent in split_sent_eval]
sents_eval = [nltk.tokenize.word_tokenize(sent) for sent in prepro_eval if len(sent) > 1]

#labeling evaluation files
tokens_eval = []
boundaries_eval = set()
offset_eval = 0
for sent in sents_eval:
    tokens_eval.extend(sent)
    offset_eval += len(sent)
    boundaries_eval.add(offset_eval-1)

#make featureset for evaluation sets
featuresets_eval = [(find_sent_finish(tokens_eval, i), (i in boundaries_eval))
               for i in range(1, len(tokens_eval)-1)
              if tokens_eval[i] in '.?!']

#segment evaluation files with no reference
split_sent_eval_no = gni_reader_eval_no.raw().split('\n')
prepro_eval_no = [sent for sent in split_sent_eval_no]
sents_eval_no = [nltk.tokenize.word_tokenize(sent) for sent in prepro_eval_no if len(sent) > 1]

#labeling evaluation files with no reference
tokens_eval_no = []
boundaries_eval_no = set()
offset_eval_no = 0
for sent in sents_eval_no:
    tokens_eval_no.extend(sent)
    offset_eval_no += len(sent)
    boundaries_eval_no.add(offset_eval_no-1)

#make featureset for evaluation sets with no reference
featuresets_eval_no = [(find_sent_finish(tokens_eval_no, i), (i in boundaries_eval_no))
               for i in range(1, len(tokens_eval_no)-1)
              if tokens_eval_no[i] in '.?!']

#evaluate with reference
print("Naive Bayes Classifier Accuracy : " + str(nltk.classify.accuracy(classifier,featuresets_eval)))
print("Decision Tree Classifier Accuracy : " + str(nltk.classify.accuracy(decisionClassifier,featuresets_eval)))
print("Max Ent Classifier Accuracy : " + str(nltk.classify.accuracy(maxentClassifier, featuresets_eval)))

#evaluate with no reference
print("Naive Bayes Classifier Accuracy : " + str(nltk.classify.accuracy(classifier,featuresets_eval_no)))
print("Decision Tree Classifier Accuracy : " + str(nltk.classify.accuracy(decisionClassifier,featuresets_eval_no)))
print("Max Ent Classifier Accuracy : " + str(nltk.classify.accuracy(maxentClassifier, featuresets_eval_no)))

#evaluate classifier accuracy with brown corpus
print("Naive Bayes Classifier Accuracy : " + str(nltk.classify.accuracy(classifier,featuresets_brown)))
print("Decision Tree Classifier Accuracy : " + str(nltk.classify.accuracy(decisionClassifier,featuresets_brown)))
print("Max Ent Classifier Accuracy : " + str(nltk.classify.accuracy(maxentClassifier, featuresets_brown)))
