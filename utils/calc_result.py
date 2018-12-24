import numpy as np
import pandas as pd

from regular_expression import is_num


class ConfusionMatrix():
    def __init__(self):
        case_types = [
            'none', 'exo1', 'exo2', 'exoX',
            'intra(dep)', 'intra(dep)_false',
            'intra(zero)', 'intra(zero)_false',
            'inter(zero)', 'inter(zero)_false'
        ]
        index = pd.MultiIndex.from_arrays([['predicted'] * 10, case_types])
        case_types = [
            'none', 'exo1', 'exo2', 'exoX',
            'intra(dep)', 'intra(zero)', 'inter(zero)'
        ]
        columns = pd.MultiIndex.from_arrays([['actual'] * 7, case_types])
        df = pd.DataFrame(data=0, index=index, columns=columns)
        self.df = df

    def calculate(self, _batch, _predict_index, target_case):
        if _predict_index == 0:
            predict_case_type = 'none'
        elif _predict_index == 1:
            predict_case_type = 'exoX'
        elif _predict_index == 2:
            predict_case_type = 'exo2'
        elif _predict_index == 3:
            predict_case_type = 'exo1'
        else:
            target_verb_index = _batch[1].name
            verb_phrase_number = _batch[0]['n文節目'][target_verb_index]
            if 'n文目' in _batch[0].keys() and _batch[0]['n文目'][target_verb_index] != _batch[0]['n文目'][_predict_index]:
                predict_case_type = 'inter(zero)'
            elif _predict_index >= len(_batch[0]):
                predict_case_type = 'inter(zero)'
            else:
                # 文内解析時（文内解析時は，文間ゼロ照応に対して予測することはありえない）
                predict_dependency_relation_phrase_number = _batch[0]['係り先文節'][_predict_index]
                if verb_phrase_number == predict_dependency_relation_phrase_number:
                    predict_case_type = 'intra(dep)'
                else:
                    predict_case_type = 'intra(zero)'

        correct_case_index_list = [int(i) for i in _batch[1][target_case].split(',')]
        for correct_case_index in correct_case_index_list.copy():
            eq = _batch[0]['eq'][correct_case_index]
            if is_num(eq):
                correct_case_index_list += np.arange(len(_batch[0]))[_batch[0]['eq'] == eq].tolist()
        if _predict_index in correct_case_index_list:
            # 予測正解時
            actual_case_type = predict_case_type
            is_correct = True
        else:
            # 予測不正解時
            actual_case_type = _batch[1][f'{target_case}_type'].split(',')[0]
            if predict_case_type == 'intra(dep)' or predict_case_type == 'intra(zero)' or predict_case_type == 'inter(zero)':
                predict_case_type += '_false'
            is_correct = False
        self.df['actual'][actual_case_type]['predicted'][predict_case_type] += 1
        return is_correct

    def calculate_f1(self):
        case_types = ['none', 'exo1', 'exo2', 'exoX', 'intra(dep)', 'intra(zero)', 'inter(zero)']
        columns = ['precision', 'recall', 'F1-score']
        f1_df = pd.DataFrame(data=0.0, index=case_types, columns=columns)
        for case_type in case_types:
            tp = self.df['actual', case_type]['predicted', case_type]
            # precision
            tp_fp = sum(self.df.loc['predicted', case_type])
            if case_type == 'intra(zero)' or case_type == 'intra(dep)' or case_type == 'inter(zero)':
                tp_fp += sum(self.df.loc['predicted', f'{case_type}_false'])
            if tp_fp != 0:
                f1_df['precision'][case_type] = tp / tp_fp
            # recall
            if sum(self.df['actual', case_type]) != 0:
                f1_df['recall'][case_type] = tp / sum(self.df['actual', case_type])
            # F1-score
            if (f1_df['precision'][case_type] + f1_df['recall'][case_type]) != 0:
                f1_df['F1-score'][case_type] = (2 * f1_df['precision'][case_type] * f1_df['recall'][case_type]) / (f1_df['precision'][case_type] + f1_df['recall'][case_type])
        all_tp = 0
        all_tp_fp = 0
        all_tp_fn = 0
        for case_type in case_types:
            all_tp += self.df['actual', case_type]['predicted', case_type]
            all_tp_fp += sum(self.df.loc['predicted', case_type])
            all_tp_fn += sum(self.df['actual', case_type])
        f1_df.loc['total'] = 0
        f1_df['precision']['total'] = all_tp / (all_tp_fp)
        f1_df['recall']['total'] = all_tp / (all_tp_fn)
        f1_df['F1-score']['total'] = (2 * f1_df['precision']['total'] * f1_df['recall']['total']) / (f1_df['precision']['total'] + f1_df['recall']['total'])
        return f1_df
