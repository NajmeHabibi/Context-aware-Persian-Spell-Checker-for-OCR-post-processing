import re
from prettytable import PrettyTable
from torch.nn import functional as F
import torch
from symspellpy.symspellpy import SymSpell, Verbosity
import jellyfish
from transformers import BertTokenizer, BertForMaskedLM
from src.models.utils import spell_checking_for_text

# Colors
R = "\033[0;31;40m"  # RED
G = "\033[0;32;40m"  # GREEN
Y = "\033[0;33;40m"  # Yellow
B = "\033[0;34;40m"  # Blue
N = "\033[0m"  # Reset
BG = "\033[0;37;42m"  # background green


class Bert_plus_SymSpell:
    def __init__(self, max_edit_distance_dictionary, prefix_length, bert_model):
        self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        self.model, self.tokenizer = self.load_bert_model(bert_model)

    def is_number(self, word):
        """
        This method checks weather a word is a number or not
        :param word: String
        :return: Boolean
        """
        new_word = re.sub("\d+", "", word)
        if len(new_word) == 0 or len(new_word) / len(word) < 0.8:
            return True
        else:
            return False

    def clean_punctuation(self, word):
        """
        This method remove the punctuations in a word in order to have cleaner text.
        :param word: String
        :return: new_word
        """
        new_word = re.sub("['?!.,؟،<)(]", "", word)
        return new_word

    def set_dictionaries(self):
        """
        In this method we set the dictionaries, in order to change them you must change the path for each dictionary.
        """
        dictionary_path = "/data/raw/fa_unigram.txt"
        bigram_path = "/data/raw/bigram2.txt"
        if not self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
            print("Dictionary file not found")
        if not self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2):
            print("Bigram dictionary file not found")

    def load_bert_model(self, bert_model):
        """
        This method load a Bert model.
        :param bert_model: name of a bert model that you want to use,
                           -for example: "HooshvareLab/bert-base-parsbert-uncased"
        :return: the loaded model with tokenizer
        """
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        model = BertForMaskedLM.from_pretrained(bert_model, return_dict=True)
        return model, tokenizer

    def get_misspelled_words_and_masked_text(self, original_text):
        """
        For a given text, this method finds the misspelled words
        and replaces them with [MASK] token in the original_text.
        This method searches each word in dictionary.
        :param original_text:  input text with misspelled words
        :return:
            -a list of misspelled words (misspelled_words),
            -masked text,
            -a list of 0&1 that determines which words are misspelled. 0 -> misspelled, 1 -> correct
        """
        misspelled_words = []
        ori_list = original_text.split()
        misspelled_words_labels = [(1 if (self.clean_punctuation(word) in self.sym_spell.words and not self.is_number(word)) else 0) for word in ori_list]
        for i, word in enumerate(ori_list):
            if not misspelled_words_labels[i]:
                misspelled_words.append(word)
                ori_list[i] = '[MASK]'
        masked_text = ' '.join(ori_list)
        return misspelled_words, masked_text, misspelled_words_labels

    def get_symspell_suggestion(self, original_text):
        """
        This method returns the symspell suggestions.
        :param original_text: input text
        :return: symspell suggestions
        """
        suggestions = self.sym_spell.lookup_compound(original_text, max_edit_distance=1)
        return suggestions

    def get_bert_suggestion_for_each_mask(self, original_text, incorrect_words_position, num_suggestions):
        """
        This function suggests the correct word for each mask in sentence.
        :param original_text: input text contains error
        :param incorrect_words_position:
        :param num_suggestions: the number of candidates that you want to have for each incorrect word
        :return: corrected sentence
        """
        ori_list = original_text.split()
        for i, pos in enumerate(incorrect_words_position):
            if not pos:
                incorrect_word = ori_list[i]
                if incorrect_word in ['.', ';', ')', '(', '،', ' ']:
                    continue
                # replace incorrect word with [MASK] token
                ori_list[i] = '[MASK]'
                masked_text = ' '.join(ori_list)
                tokenized_text = self.tokenizer.tokenize(masked_text)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                MASKID = [i for i, e in enumerate(tokenized_text) if e == '[MASK]'][0]
                # Create the segments tensors
                softmax = self.get_softmax_layer(tokenized_text, indexed_tokens)
                mask_word = softmax[0, MASKID, :]
                top_n = torch.topk(mask_word, num_suggestions)
                candidates = [self.tokenizer.decode([token]) for token in top_n.indices]
                values = [round(value.item(), 4) for value in top_n.values]
                candidates_dictionary = {k: v for k, v in zip(candidates, values)}
                # get the best candidate
                similar_suggestion, similar_suggestion_score = self.get_top_similar_suggestion(candidates_dictionary, incorrect_word)
                if similar_suggestion is not None:
                    ori_list[i] = similar_suggestion
                else:
                    ori_list[i] = incorrect_word
                # you can use SymSpell suggestions as well!
                symspell_suggestions = self.sym_spell.lookup(incorrect_word, Verbosity.CLOSEST, include_unknown=True)
        result = ' '.join(ori_list)
        return result

    def get_softmax_layer(self, tokenized_text, indexed_tokens):
        """
        This method use pre-trained model to predict candidates
        :param tokenized_text: bert tokenizer output for each sentence
        :param indexed_tokens: index of tokens
        :return: the softmax layer
        """
        # Create the segments tensors
        segs = [i for i, e in enumerate(tokenized_text) if e == "."]
        segments_ids = []
        prev = -1
        for k, s in enumerate(segs):
            segments_ids = segments_ids + [k] * (s - prev)
            prev = s
        segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
        segments_tensors = torch.tensor([segments_ids])
        # prepare Torch inputs
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        logits = predictions.logits
        softmax = F.softmax(logits, dim=-1)
        return softmax

    def levenshtein_distance(self, s1, s2):
        """
        This method compute the levenshtein distance between two string
        :param s1: first word
        :param s2: second word
        :return: (int) levenshtein_distance
        """
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances

    def get_top_similar_suggestion(self, candidates_list, misspelled_word):
        """
        This method selects the best and most similar candidate for misspelled word.
        :param candidates_list:  a list of tuples which are the bert_suggested candidates for the misspelled word
        and their bert output scores, for example: [ ("Comprehensive", 0.82), ("Universal", 0.4), ...]
        :param misspelled_word:  misspelled word
        :return:  most_similar_suggestion with most_similar_suggestion_score for example: ("Comprehensive", 120)
        """
        # this is a table which contains information for print
        table = PrettyTable(['Bert_score', 'Edit_distance', 'Incorrect_word', 'Suggested'])
        most_similar_suggestion = None
        most_similar_suggestion_score = 0

        # for each candidate we check the bert output score and levenshtein_distance
        for suggested in candidates_list:
            # calculate the levenshtein_distance
            distance = jellyfish.levenshtein_distance(suggested, misspelled_word)
            # bert score
            bert_score = round(candidates_list[suggested], 3)
            if distance == 0:
                table.add_row([G + suggested + N, misspelled_word, round(distance, 3), bert_score])
                most_similar_suggestion = suggested
            else:
                # final score for each candidate
                score = 1 / distance * 100 + bert_score * 50
                if score > most_similar_suggestion_score and score > 100:
                    table.add_row([G + suggested + N, misspelled_word, round(distance, 3), bert_score])
                    most_similar_suggestion = suggested
                    most_similar_suggestion_score = score
                else:
                    table.add_row(
                        [N + suggested, misspelled_word, round(distance, 3), bert_score])
        # if you want to print the table, uncomment the line below
        # print(table)
        return most_similar_suggestion, most_similar_suggestion_score


if __name__ == "__main__":

    spell_checker = Bert_plus_SymSpell(max_edit_distance_dictionary=2, prefix_length=7,
                          bert_model='HooshvareLab/bert-base-parsbert-uncased')
    spell_checker.set_dictionaries()
    # spell_checking_for_csv(spell_checker, data_path="/Users/najme/projects/spell_checker/data/raw/data100.csv")
    corrected_text = spell_checking_for_text(spell_checker, "واقعیت یک متن باید نابل درک باشد، خواه به وشیله حواس "
                                                           "ماننذ درک خطوط، اشکال و ایماها و اشاره‌ها یا ادراک "
                                                            "پدیدار شناسانه مانند ادراک تصورات ذهنی.")
    print(corrected_text)