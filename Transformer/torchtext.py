from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Test translation
def translate(sentence):
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example translation
test_sentence = "I am a student."
print("Translated Sentence:", translate(test_sentence))
