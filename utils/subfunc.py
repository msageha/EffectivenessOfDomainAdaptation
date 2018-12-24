import numpy as np
import pandas as pd
from regular_expression import is_num


def return_file_domain(file):
    domain_dict = {'PM':'雑誌','PN':'新聞', 'OW':'白書', 'OC':'Yahoo!知恵袋', 'OY':'Yahoo!ブログ', 'PB':'書籍'}
    for domain in domain_dict:
        if domain in file:
            return domain
    raise ValueError(f'Unknown file name {file}')


def predicted_log(batch, pred, target_case, corrects):
    batchsize = len(batch)
    for i in range(batchsize):
        target_verb_index = batch[i][1].name
        predicted_argument_index = pred[i].item()
        actual_argument_index = int(batch[i][1][target_case].split(',')[0])
        target_verb = batch[i][0]['単語'][target_verb_index]
        if predicted_argument_index >= len(batch[i][0]):
            predicted_argument = 'inter(zero)'
        else:
            predicted_argument = batch[i][0]['単語'][predicted_argument_index]

        actual_argument = batch[i][0]['単語'][actual_argument_index]
        sentence = ' '.join(batch[i][0]['単語'][4:])
        file = batch[i][2]
        log = {
            '正解': corrects[i],
            '述語位置': target_verb_index - 4,
            '述語': target_verb,
            '正解項位置': actual_argument_index - 4,
            '正解項': actual_argument,
            '予測項位置': predicted_argument_index - 4,
            '予測項': predicted_argument,
            '解析対象文': sentence,
            'ファイル': file
        }
        domain = return_file_domain(file)
        yield domain, log
