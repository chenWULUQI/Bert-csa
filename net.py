from transformers import BertModel
import torch
import torch.nn as nn


device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
pretrained=BertModel.from_pretrained("bert-base-chinese").to(device)
# 整体模型十分简单，先用预训练的Bert模型提取文本中的特征，再用一个全连接层将提取到的768维特征映射到2维以便后续情感识别任务的二分类需求。 
# 三种取出模型某一层网络的方法 1.print(pretrained.embeddings)2.print(pretrained[0])3.print(pretrained["embeddings"])

class ClassifyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(768,2)
    def forward(self,input_ids,attention_mask,token_type_ids):
        with torch.no_grad():
            out=pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # print(out.shape)    
        out=self.fc(out.last_hidden_state[:,0])
        # print(out.shape)   
        out=out.softmax(dim=1)
        return out
    

