from transformers import pipeline
nlp = pipeline("sentiment-analysis")
result = nlp("I love this!")
print(result)