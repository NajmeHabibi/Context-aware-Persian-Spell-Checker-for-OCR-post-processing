![spell checker](https://user-images.githubusercontent.com/42568062/174132703-e5afc1c2-8076-4493-8706-5a8c657f0404.png)


## This is a Context-aware Persian Spell Checker for OCR post-processing

### Introduction
Spell Correction is an important and at the same time challenging task because the optimal result requires the ability to understand language at the human level. In this project, we mainly focus on correcting OCR outputs which extract text from an image or Pdf. There are many different ways to design a spell checker. We go for dictionary-based and machine learning methods. Context-aware spell checking is the most important feature of our proposed model. In fact, this model is able to correct spelling errors according to the general meaning of the sentence by using BERT model. 

## Results
| DataSet | Detection Accuracy | Correction Accuracy|
|---------|--------------------|--------------------|
| `10000 Sentence on Persian WikiPedia` | %76.7 | %69.3|

### - How to run?
1. For Spell Checking run **Bert-spellchecker.py**, in the main function:
   * **Spell_checking_for_csv** function does spell checking for each row in CSV file
   * **Spell_checking_for_text** function does spell checking just for an input text
2. In order to create a CSV data for test and train run **data_creator.py**

### - How does It work?
1. For a given text spell checker finds misspelled words by **get_misspelled_words_and_masked_text** method,
   * In this method original text split into words (by " ")
   * Each word get checked in dictionary,
   * if a word is  out of dictionary, it's a misspelled word 
2. Then we pass the misspelled words position and the original text to **get_bert_suggestion_for_each_mask** method, in order to correct the text,
3. In **get_bert_suggestion_for_each_mask** method:
   * The original text split into words (by " "),  
   * Then each misspelled word replace with [MASK] token in text separately,
   * The masked text passes to **BERT tokenizer**,
   * After preprocessing the tokenized_text we get the softmax layer by calling **get_softmax_layer** method, 
   * Softmax layer contains all suggestions for [MASK] token, we find the top N suggestions, calling **torch.topk**,
   * Then we decode the suggested ids to words by **Bert tokenizer.decode** method, this method returns the candidates, 
   * Then to get the best match for misspelled word we call **get_top_similar_suggestion**:
     * This method computes **Levenshtein Distance** (jellyfish.levenshtein_distance),
     * Merges the **BERT probability** score with **Levenshtein Distance** by this formula: score = 1 / distance * 100 + bert_score * 50,
     * Then returns the word with the highest score as the best candidate (in some cases, it returns None, which means none of the candidates are appropriate)
   * Finally, if **get_top_similar_suggestion** method returns the best one: [MASK] token will be replaced with the best candidate, else it will be replaced with the misspelled word. 

