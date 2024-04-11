import json
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from tqdm import tqdm
from datasets import load_dataset
import re

model_directory = "/kaggle/input/final-model"
paraphrase_model = T5ForConditionalGeneration.from_pretrained(model_directory).to("cuda")
paraphrase_tokenizer = T5TokenizerFast.from_pretrained(model_directory)


def paws_text_preprocess(text):
    text = re.sub(r'\s+([.,;:-?!])', r'\1', text)
    text = re.sub(r'\(\s+', r'(', text)
    text = re.sub(r'\s+\)', r')', text)
    return text


def process_paws_dataset(paws_data):
    processed_paws_dataset = []
    for instance in paws_data:
        if instance["label"] == 1:
            processed_paws_dataset.append(
                {"Text": paws_text_preprocess(instance["sentence1"]), "Paraphrase": paws_text_preprocess(instance["sentence2"])})
    return processed_paws_dataset


paws_test_dataset = load_dataset("paws", "labeled_final")["test"]
paws_test_data = process_paws_dataset(paws_test_dataset)

test_data_path = "/kaggle/working/final_test.json"
file = open(test_data_path, "r", encoding="utf-8")
test_data = json.load(file)
file.close()


def paraphrase_sentences(data, model, tokenizer, batch_size=16):
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc="Generating paraphrases"):
        batch = data[i:i+batch_size]
        original_sentences = []
        for instance in batch:
            original_sentences.append(instance["Text"])
        tokenized_sentences = tokenizer(original_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            encoded_sentences = model.generate(
                **tokenized_sentences,
                max_length=512,
                do_sample=True,
                top_k=50,
                top_p=0.95)
        paraphrases = []
        for output in encoded_sentences:
            paraphrase = tokenizer.decode(output, skip_special_tokens=True)
            paraphrases.append(paraphrase)
        reference_paraphrases = []
        for instance in batch:
            reference_paraphrases.append(instance["Paraphrase"])
        for original_sentence, paraphrase, reference in zip(original_sentences, paraphrases, reference_paraphrases):
            results.append({"Text": original_sentence, "Reference_Paraphrase": reference, "Generated_Paraphrase": paraphrase})
    return results


paraphrase_results = paraphrase_sentences(paws_test_data, paraphrase_model, paraphrase_tokenizer)

results_path = "/kaggle/working/paraphrase_results.json"
file = open(results_path, "w", encoding="utf-8")
json.dump(paraphrase_results, file, indent=4)
file.close()
