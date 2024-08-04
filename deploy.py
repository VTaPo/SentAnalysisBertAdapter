import streamlit as st
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from adapters import AdapterConfig
from adapters import BertAdapterModel


# Predict function
def predict(data, model, tokenizer):
    inputs = tokenizer(data['text'], padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    mask = inputs['attention_mask'].to(device)
    input_id = inputs['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)[0]
    probabilities = torch.nn.functional.softmax(output, dim=1)

    class_0_prob = probabilities[0][0].item()
    class_1_prob = probabilities[0][1].item()
    return class_0_prob, class_1_prob

# BERT Class
class BertClassifierwithAdapter(nn.Module):
    def __init__(self, model_id='bert-base-cased', adapter_id='pfeiffer',
                 task_id = 'sentiment_analysis', num_class=2):
        super(BertClassifierwithAdapter, self).__init__()
        self.adapter_config = AdapterConfig.load(adapter_id)
        self.bert = BertAdapterModel.from_pretrained(model_id)
        # Insert adapter according to configuration
        self.bert.add_adapter(task_id, config=self.adapter_config)
        # Freeze all BERT-base weights
        self.bert.train_adapter(task_id)
        # Add prediction layer on top of BERT-base
        self.bert.add_classification_head(task_id, num_labels=num_class)
        # Make sure that adapters and prediction layer are used during forward pass
        self.bert.set_active_adapters(task_id)
        
    def forward(self, input_id, mask):
        output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        return output

# Load model and tokenizer
tokenizer_model_name = 'bert-base-cased'
model_adapter_path = 'model/bert_adapter/' 

tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
task_name = 'sentiment_analysis'

model_path = f'{model_adapter_path}{task_name}'

# Load trained adapter
trained_adapter_model = BertClassifierwithAdapter(task_id=task_name)
adapter_name = trained_adapter_model.bert.load_adapter(model_path)
trained_adapter_model.bert.set_active_adapters(adapter_name)
trained_adapter_model.to(device)
trained_adapter_model.eval()

# Streamlit UI
st.title('SENTIMENT ANALYSIS')
st.write('For English and Vietnamese sentences')

text_input = st.text_area('Enter your text here:')
compute_button = st.button('Compute')

if compute_button and text_input:
    data = {'text': text_input}
    class_0_prob, class_1_prob = predict(data, trained_adapter_model, tokenizer)

    # PLot pie chart
    labels = ['Negative', 'Positive']
    probabilities = [class_0_prob, class_1_prob]
    colors = sns.color_palette('pastel')[0:2]
    
    fig, ax = plt.subplots()
    ax.pie(probabilities, labels=[f"{label} {prob*100:.2f}%" for label, prob in zip(labels, probabilities)], colors=colors, autopct='%1.1f%%')
    ax.legend(labels, loc="best")
    ax.set_title('Sentiment Analysis Result')
    st.pyplot(fig)