from torch.utils.data import Dataset
from datasets import load_from_disk
from datasets import load_dataset                         ####### 此方法为hugging face平台提供，可直接在线访问官网数据集，无需下载到本地使用






class Mydataset(Dataset):
    def __init__(self,mode):
        super().__init__()
        self.dataset=load_from_disk(r"D:\cccc\usense_work\hgf_test\data\lansinuote___chn_senti_corp\data_csc")
        if mode=="train":
            self.dataset=self.dataset["train"]
        elif mode=="validation":
            self.dataset=self.dataset["validation"]
        elif mode=="test":
            self.dataset=self.dataset["test"]
        else:
            print("数据集加载错误")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        text=self.dataset[index]["text"]
        label=self.dataset[index]["label"]
        return text,label
    

if __name__=='__main__':
    dataset=Mydataset("validation")
    for i, data in enumerate(dataset):
        print(data)
        if i >= 5:  
            break
        