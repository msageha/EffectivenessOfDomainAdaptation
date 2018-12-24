import json
import os
import torch


def dump_dict(dic, dump_dir, file_name):
    os.makedirs(f'./{dump_dir}/', exist_ok=True)
    with open(f'./{dump_dir}/{file_name}.json', 'w') as f:
        json.dump(dic, f, indent=2)


def dump_predict_logs(logs, dump_dir):
    os.makedirs(f'./{dump_dir}/predicts/', exist_ok=True)
    for domain in logs.keys():
        with open(f'{dump_dir}/predicts/{domain}.tsv', 'w') as f:
            f.write('\t'.join(logs[domain][0].keys()))
            f.write('\n')
            for log in logs[domain]:
                values = [str(value) for value in log.values()]
                f.write('\t'.join(values))
                f.write('\n')


def save_model(epoch, model, dump_dir, gpu):
    print('--- save model ---')
    os.makedirs(f'./{dump_dir}/model/', exist_ok=True)
    model.cpu()
    torch.save(model.state_dict(), f'./{dump_dir}/model/{epoch}.pkl')
    if gpu >= 0:
        model.cuda()
