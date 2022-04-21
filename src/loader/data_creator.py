import os
import random
import re
import pandas as pd
from hazm import Normalizer
import stanza
from parsivar import Normalizer
import csv

ALEF = 'ا'
BE = 'ب'
PE = 'پ'
TE = 'ت'
CE = 'ث'
JIM = 'ج'
CHE = 'چ'
HE = 'ح'
KHE = 'خ'
DAL = 'د'
ZAL = 'ذ'
RE = 'ر'
ZE = 'ز'
SIN = 'س'
SHIN = 'ش'
SAD = 'ص'
ZAD = 'ض'
TA = 'ط'
ZA = 'ظ'
AIN = 'ع'
GHAIN = 'غ'
FE = 'ف'
GHAF = 'ق'
KAF = 'ک'
GAF = 'گ'
LAM = 'ل'
MIM = 'م'
NOON = 'ن'
VAV = 'و'
HE1 = 'ه'
YE = 'ی'
DELETE = ''
persian_characters = [ALEF, BE, PE, TE, CE, JIM, CHE, HE, KHE, DAL, ZAL, RE, ZE, SIN, SHIN, SAD, ZAD, TA, ZA, AIN,
                      GHAIN, FE, GHAF, KAF, GAF, LAM, MIM, NOON, VAV, HE1, YE]
MAX_LEN_SENT = -1


def create_misspell_error(correct_word):
    my_rand = random.randint(0, 100)
    if my_rand < 20:
        index = random.randint(0, len(correct_word) - 1)
        index_alephba = random.randint(0, len(persian_characters) - 1)
        while persian_characters[index_alephba] == correct_word[index]:
            index = random.randint(0, len(correct_word) - 1)
            index_alephba = random.randint(0, len(persian_characters) - 1)
        misspell_word = correct_word.replace(correct_word[index], persian_characters[index_alephba], 1)
    else:
        index = random.randint(0, len(correct_word) - 1)
        if correct_word[index] in alphabet_prob_missing_dict:
            prob_list = alphabet_prob_missing_dict[correct_word[index]]
            index_alpha = random.randint(0, len(prob_list) - 1)
            misspell_word = correct_word.replace(correct_word[index], prob_list[index_alpha], 1)
        else:
            misspell_word = correct_word.replace(correct_word[index], '', 1)

    return misspell_word


def is_persian_word(word):
    print(word)
    for ch in word:
        if not (u'\u0600' <= ch <= u'\u06FF'):
            print(False)
            return False
    print(True)
    return True


def create_misspell_list(words):
    label_list = []
    is_correct = True
    correct_list = [word.text for word in words]
    tokens = correct_list.copy()
    for i, token in enumerate(tokens):
        if not is_persian_word(token):
            label_list.append(str(0))
            continue
        prob = random.random()
        if prob < 0.15:
            misspell_word = create_misspell_error(token)
            tokens[i] = misspell_word
            label_list.append(str(1))
            is_correct = False
        else:
            label_list.append(str(0))

    return tokens, correct_list, label_list, is_correct


def create_n_word_lists(article_text):
    article_text = article_text.replace('\u200c', ' ')
    data_list = []
    for sentence in article_text.split('.'):
        if 3 < len(sentence):
            sentence = sentence.rstrip() + '.'
            sentence = sentence.lstrip()
            doc = nlp(sentence)
            words = doc.sentences[0].words
            num_words = len(words)
            if 3 < num_words <= 20:
                processed_data = create_misspell_list(words)
                data_list.append(processed_data)
    return data_list


def create_data(limit):
    counter = 0
    df = pd.DataFrame(columns=['random_text', 'origin_text', 'label', 'is_correct'])
    for article in os.listdir("/data/processed/all-articles"):
        with open(f"/data/processed/all-articles/{article}", "r") as f:
            article_text = f.read()
            f.close()
            data_list = create_n_word_lists(article_text)
            for data in data_list:
                random_text, origin_text, label, is_correct = data
                output = {'random_text': random_text,
                          'origin_text': origin_text,
                          'label': label,
                          'is_correct': is_correct}
                df = df.append(output, ignore_index=True)
                counter += 1
                if counter % 100 == 0:
                    print(counter)
        if counter % 10000 == 0:
            df.to_csv(f"/data/processed/data{limit}.csv")
    df.to_csv(f"/data/processed/data{limit}.csv")


def text_preprocessing(text):
    text = re.sub(
        '[^\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC\u200cئؤإأء؛ ?؟!ًٌٍَُِّ,. :،ُ)ـ(\d+]',
        " ", text)
    text = text.replace("()", " ")
    text = re.sub("\s+", " ", text)

    normalizer = Normalizer()
    text = normalizer.normalize(text)
    text = text.lstrip()
    text = text.rstrip()
    return text


def create_txt_files():
    counter = 1
    for wikitext in os.listdir("/data/raw"):
        print(wikitext)
        with open(f"/data/raw/{wikitext}", 'r') as reader:
            text = reader.read()
            for article in text.split("عنوان مقاله"):
                if len(article):
                    with open(f"/data/processed/all-articles/article{counter}.txt",
                              "w") as f:
                        f.write(text_preprocessing(article))
                        f.close()
                        counter += 1
                        print(counter)
            reader.close()


def create_prob_missing_char_dictionary():
    dict_from_csv = {}
    with open("/data/confusion table.csv", mode='r') as inp:
        reader = csv.reader(inp)
        for row in reader:
            i = 1
            my_list = []
            while row[i] != '':
                my_list.append(eval(row[i]))
                i += 1
                if i > len(row) - 1:
                    break
            dict_from_csv[eval(row[0])] = my_list
    return dict_from_csv


if __name__ == '__main__':
    alphabet_prob_missing_dict = create_prob_missing_char_dictionary()
    nlp = stanza.Pipeline(lang='fa', processors='tokenize', tokenize_batch_size=32)
    create_data(limit=30000)
