import json
import os


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)


def split_data(input_file, target_dir, num_train=7000, num_test=1000):
    with open(input_file, encoding='utf8') as f:
        data = json.load(f)

    train = data[:num_train]
    dev = data[num_train:num_train+num_test]
    test = data[num_train+num_test:]
    save_json(train, os.path.join(target_dir, 'train.json'))
    save_json(test, os.path.join(target_dir, 'test.json'))
    save_json(dev, os.path.join(target_dir, 'dev.json'))


if __name__ == '__main__':
    import os
    os.mkdir('data')
    os.mkdir('data/funcup_tw')
    split_data('total.json', 'data/funcup_tw')
