import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Carregar o tokenizer e o modelo pré-treinado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # assumindo três classes: positivo, neutro, negativo

# Inicializar a pipeline de classificação
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Função para classificar o sentimento
def classify_sentiment(text):
    result = nlp(text)
    return result

# Exemplo de uso
text = "I love using Python for data science!"
sentiment = classify_sentiment(text)
print(f'Text: {text}\nSentiment: {sentiment}')
