from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "yashpwr/resume-ner-bert-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

tokenizer.save_pretrained("./skill_model")
model.save_pretrained("./skill_model")
