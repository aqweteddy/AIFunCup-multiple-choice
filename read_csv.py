import pandas as pd


def read_csv(file,):
    # ID,文章,問題,選項1,選項2,選項3,選項4,正確答案
    df = pd.read_csv(file)
    file = file.split('/')[-1].split('.csv')[0]
    return [{'id': str(i+1),
             'content': item[1][1],
             'question': item[1][2],
             'op1': item[1][3],
             'op2': item[1][4],
             'op3': item[1][5],
             'op4': item[1][6],
             'ans': str(item[1][7]) if len(str(item[1][7])) == 1 else str(item[1][7])[2],
             } for i, item in enumerate(df.iterrows())]


if __name__ == '__main__':
    import os
    import json

    # result = []
    result = read_csv('./corpus/test_data.csv')
    # for root, dirs, files in os.walk("./corpus/", topdown=True):
    #     files = sorted(files)
    #     for name in files:
    #         if '.csv' in name and 'inter' not in name:
    #             print(os.path.join(root, name))
    #             result += read_csv(os.path.join(root, name))

    with open('data/test_data/test.json', 'w') as f:
        print(len(result))
        json.dump(result, f, ensure_ascii=False)

