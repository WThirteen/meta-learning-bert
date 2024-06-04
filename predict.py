import torch  
from transformers import BertTokenizer
def load_model():

    model = torch.load('model_files/epoch0.pth')  

    tokenizer = BertTokenizer.from_pretrained('E:/hugging_face_model_all/bert-base-uncased', do_lower_case = True)

    return model, tokenizer

def enter_text():

    print("text:")
    print("---------------------------------")

    text = input()

    return text

def pre(model,inputs):

    model.eval()  
  
    # 禁用梯度计算以节省显存和计算资源  
    with torch.no_grad():  
        # 获取模型的输出（logits）  
        outputs = model(**inputs)  
        logits = outputs.logits  
  
    # 应用softmax函数获取概率分布（可选）  
    probs = torch.nn.functional.softmax(logits, dim=1)  
  
    # 获取预测类别（概率最高的类别）  
    _, predicted_class = torch.max(probs, 1)  

    id = predicted_class.item()

    LABEL_MAP  = {'positive':0, 'negative':1, 0:'positive', 1:'negative'}

    res = LABEL_MAP[id]
  
    # 打印预测结果  
    print("---------------------------------")
    print(f"Predicted class: {res}")

    return res



def run():
    model, tokenizer = load_model()
    text = enter_text()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)  

    # 确保模型在正确的设备上（比如GPU，如果可用）  
    if torch.cuda.is_available():  
        model.to('cuda')  
        inputs['input_ids'] = inputs['input_ids'].to('cuda')  
        inputs['token_type_ids'] = inputs['token_type_ids'].to('cuda')
        inputs['attention_mask'] = inputs['attention_mask'].to('cuda')  

    res = pre(model,inputs)


if __name__ == "__main__":
    run()
