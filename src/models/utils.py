import pandas as pd


def spell_checking_for_text(spell_checker, text):
    """
    This function corrects the given text
    :param spell_checker: the pre-trained model
    :param text: misspelled text
    :return: corrected text
    """
    incorrect_words, masked_text, incorrect_words_position = spell_checker.get_misspelled_words_and_masked_text(text)
    result = spell_checker.get_bert_suggestion_for_each_mask(text, incorrect_words_position, num_suggestions=10)
    return result


def spell_checking_for_csv(spell_checker, data_path):
    """
    This function corrects all sentences in a csv file and print the metrics
    :param spell_checker: the pre-trained model
    :param data_path: location of data
    """
    data = pd.read_csv(data_path)
    DTP, DFP, DTN, DFN = [0] * 4
    CTP, CFP, CTN, CFN = [0] * 4
    for index, row in data.iterrows():
        random_list = eval(row["random_text"])
        random_text = ' '.join(random_list)
        origin_text = ' '.join(eval(row["origin_text"]))
        label = eval(row["label"])

        incorrect_words, masked_text, incorrect_words_position = spell_checker.get_misspelled_words_and_masked_text(
            random_text)

        TP, FP, TN, FN = detection_metrics(incorrect_words_position, label)
        DTP += TP
        DFP += FP
        DTN += TN
        DFN += FN

        result = spell_checker.get_bert_suggestion_for_each_mask(random_text, incorrect_words_position,
                                                                 num_suggestions=10)
        TP2, FP2, TN2, FN2 = correction_metrics(result, origin_text, incorrect_words_position)
        CTP += TP2
        CFP += FP2
        CTN += TN2
        CFN += FN2

    CRecall, CPrecision, CF1, CAccuracy = get_all_metrics(CTP, CTN, CFP, CFN)
    print("Correction Results:")
    print("recall:", CRecall, " precision:", CPrecision, " f1:", CF1, " accuracy:", CAccuracy)

    DRecall, DPrecision, DF1, DAccuracy = get_all_metrics(DTP, DTN, DFP, DFN)
    print("Detection Results:")
    print("recall:", DRecall, " precision:", DPrecision, " f1:", DF1, " accuracy:", DAccuracy)


def get_all_metrics(TP, TN, FP, FN):
    """
    This function returns the final accuracy results of out model
    :param TP: an outcome where the model correctly predicts the positive class
    :param TN: an outcome where the model correctly predicts the negative class
    :param FP: an outcome where the model incorrectly predicts the positive class
    :param FN: an outcome where the model incorrectly predicts the negative class
    :return:
    """
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return recall, precision, f1_score, accuracy


def correction_metrics(prediction, labels, detection_label):
    """
    This function compute  TP, FP, TN, FN in each corrected sentence in correction part
    :param prediction: the corrected sentence by our model
    :param labels: the correct sentence
    :param detection_label:
    :return:
    """
    detection_label = [1 - x for x in detection_label]
    prediction_list = prediction.split()
    labels_list = labels.split()
    TP, FP, TN, FN = [0] * 4
    # T, F = [0] * 2
    for ms, predict, label in zip(detection_label, prediction_list, labels_list):
        predict = predict.replace(" ", "")
        label = label.replace(" ", "")
        is_correct = predict == label
        if ms:  # misspell
            TP += is_correct
            FN += not is_correct
        else:
            TN += is_correct
            FP += not is_correct
    return TP, FP, TN, FN


def detection_metrics(prediction, labels):
    """
    This function compute  TP, FP, TN, FN in each corrected sentence in detection part
    :param prediction: a list of 0,1 which is predicted by out model
    :param labels: a list of 0,1 which is the label of out data
    :return: TP, FP, TN, FN
    """
    prediction = [1 - x for x in prediction]
    TP, FP, TN, FN = [0] * 4
    for predict, label in zip(prediction, labels):
        if label == 1:
            if predict == label:
                TP += 1
            else:
                FN += 1
        else:
            if predict == label:
                TN += 1
            else:
                FP += 1
    return TP, FP, TN, FN
