import torch
from dataset import Mydataset
from torch.utils.data import DataLoader
from net import ClassifyModel
from transformers import BertTokenizer
from torch.optim import AdamW

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
EPOCH=5

token=BertTokenizer.from_pretrained("bert-base-chinese")
def collate_fn(data):
    sentences= [i[0] for i in data]
    labels= [i[1] for i in data]
    #编码
    data=token.batch_encode_plus(
        batch_text_or_text_pairs=sentences,
        truncation=True,
        max_length=50,
        padding="max_length",
        return_tensors="pt",
        return_length=True    
    )
    input_ids=data["input_ids"]
    attention_mask=data["attention_mask"]
    token_type_ids=data ["token_type_ids"]
    labels=torch.LongTensor(labels)

    return input_ids,attention_mask,token_type_ids,labels

train_dataset=Mydataset("train")

train_loader=DataLoader(
    dataset =train_dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
    num_workers=0 
)

if __name__=='__main__':
    print(device)
    model=ClassifyModel().to(device)
    optimizer=AdamW(model.parameters(),lr=5e-4)
    loss_func=torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            
            out=model(input_ids,attention_mask,token_type_ids)
            loss=loss_func(out,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%5 ==0:
                out=out.argmax(dim=1)
                acc=(out == labels) .sum().item()/len(labels)
                print(epoch,i,loss.item(),acc)
               
        torch.save(model.state_dict(),f"checkpoints/{epoch}bert.pt")
        print(epoch,"参数保存成功")