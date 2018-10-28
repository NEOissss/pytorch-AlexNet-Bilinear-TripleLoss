import re


def analyze_log(filename):
    with open(filename, 'r') as fp:
        content = fp.read()
        match_param = re.search(r'parameters\ssaved:\s\w-\w-\w+', content)
        match_train = re.search(r'train_stats_\d+\.npy', content)
        match_test = re.search(r'test_result_\d+\.npy', content)
        match_test_accu = re.search(r'Test\saccuracy:\s\d\.\d+', content)
        match_net = re.search(r'Net:\s\w+', content)
        match_margin = re.search(r'Margin:\s\d+\.\d+', content)
        match_freeze = re.search(r'Freeze\smode:\s\w+', content)
        match_epoch = re.search(r'#Epoch:\s\d+', content)
        match_batch = re.search(r'#Batch:\s\d+', content)
        match_lr = re.search(r'rate:\s\d+\.\d+', content)

    try:
        res = {'net': match_net.group().split()[-1], 'margin': match_margin.group().split()[-1]}
        if match_freeze:
            res['freeze'] = match_freeze.group().split()[-1]
        else:
            res['freeze'] = 'None'
        res['epoch'] = match_epoch.group().split()[-1]
        res['batch'] = match_batch.group().split()[-1]
        res['lr'] = match_lr.group().split()[-1]
        res['test_accu'] = match_test_accu.group().split()[-1][:5]
        res['param'] = match_param.group().split()[-1] if match_param else None
        res['train'] = match_train.group() if match_train else None
        res['test'] = match_test.group() if match_test else None
        return res
    except AttributeError:
        return None
