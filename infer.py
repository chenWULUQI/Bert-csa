import torch
from net import ClassifyModel
from transformers import BertTokenizer

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
names=["negative","positive"]

token=BertTokenizer.from_pretrained("bert-base-chinese")

def collate_fn(data):
    sentences= []
    sentences.append(data)

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
    

    return input_ids,attention_mask,token_type_ids


def test():
    print(device)
    model=ClassifyModel().to(device)
    model.load_state_dict(torch.load("checkpoints/4bert.pt"))
    model.eval()
    # optimizer=AdamW(model.parameters(),lr=5e-4)
    # loss_func=torch.nn.CrossEntropyLoss()
    while True:
        data=input("请输入测试数据(输入“q”退出):")
        if data=="q":
            print("测试结束")
            break
        input_ids, attention_mask,token_type_ids=collate_fn(data)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            out=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            out=out.argmax(dim=1)
            print("模型判定:",names[out],"\n")
        
if __name__=='__main__' :
    test()   
   