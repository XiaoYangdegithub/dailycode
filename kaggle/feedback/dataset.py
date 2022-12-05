import torch
from torch.utils.data import Dataset #利用继承自Dataset的类，可以访问训练所需的数据
from transformers import DataCollatorForWholeWordMask
from cfg import CFG


class TrainDataset(Dataset):

    def __init__(self, df, tokenizer): #初始化创建对象，df(datafree)-训练数据，
        df.reset_index(inplace=True)
        self.labels = df[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].values
        self.texts = df['full_text'].values
        self.tokenizer = tokenizer #将句子分词并转化为唯一编码

    def __len__(self):  #定义当被len()函数调用时的行为(返回容器中的元素个数)
        # 像序列类型（如列表，元组和字符串）或映射类型（如字典）都属于容器类型
        return len(self.labels) #返回一个长度，显示迭代器中有多少labels

    def __getitem__(self, idx):  #定义获取容器中指定元素的行为，相当于self[key]，即允许类对象可以有索引操作
        text = self.texts[idx]
        label = self.labels[idx]
        output_ids = self.tokenizer(text,
                                    padding='max_length', max_length=CFG.max_position_embeddings, truncation=True)
             #padding是如果不是最大长度就要补成最大长度，truncation设为true是因为如果超过最大长度要进行截断
        return {'input_ids': torch.as_tensor(output_ids['input_ids'], dtype=torch.long),
                'attention_mask': torch.as_tensor(output_ids['attention_msk'], dtype=torch.long),
                'labels': torch.as_tensor(label, dtype=torch.float)}



if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
    df = pd.read_csv('train.csv')
    train_dataset = TrainDataset(df, tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test = next(iter(train_loader))
    for key, value in test.items():
        print(key)
        print(value.shape)
    print('done')
