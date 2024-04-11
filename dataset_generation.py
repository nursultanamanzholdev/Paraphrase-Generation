from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm
from datasets import load_dataset
import re


en_to_ru_model_name = "Helsinki-NLP/opus-mt-en-ru"
ru_to_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
en_to_ru_model = AutoModelForSeq2SeqLM.from_pretrained(en_to_ru_model_name).to("cuda")
ru_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(ru_to_en_model_name).to("cuda")

en_to_ru_tokenizer = AutoTokenizer.from_pretrained(en_to_ru_model_name)
ru_to_en_tokenizer = AutoTokenizer.from_pretrained(ru_to_en_model_name)
sentence_model = SentenceTransformer("paraphrase-MiniLM-L3-v2").to("cuda")


def process_translation(text, src_model, src_tokenizer, tgt_model, tgt_tokenizer):
    src_tokenizer.src_lang = "en"
    encoded_input = src_tokenizer(text, return_tensors="pt", max_length=60, truncation=True, padding=True).to("cuda")
    translated_output = src_model.generate(**encoded_input)
    translated_text = src_tokenizer.batch_decode(translated_output, skip_special_tokens=True)

    tgt_tokenizer.src_lang = "ru"
    encoded_back_input = tgt_tokenizer(translated_text, return_tensors="pt", max_length=60, truncation=True, padding=True).to("cuda")
    back_translated_output = tgt_model.generate(**encoded_back_input)
    back_translated_text = tgt_tokenizer.batch_decode(back_translated_output, skip_special_tokens=True)

    return back_translated_text


def compute_cosine_similarity(original, paraphrase):
    original_embedding = sentence_model.encode(original, convert_to_tensor=True)
    paraphrase_embedding = sentence_model.encode(paraphrase, convert_to_tensor=True)
    return util.pytorch_cos_sim(original_embedding, paraphrase_embedding).item()


def paraphrase_sentences(sentences, similarity_threshold=0.85):
    paraphrases = []
    for sentence in tqdm(sentences):
        back_translated = process_translation(sentence, en_to_ru_model, en_to_ru_tokenizer, ru_to_en_model, ru_to_en_tokenizer)
        if back_translated:
            cosine_similarity = compute_cosine_similarity(sentence, back_translated[0])
            if cosine_similarity > similarity_threshold:
                paraphrases.append({"Text": sentence, "Paraphrase": back_translated[0]})
    return paraphrases


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


def process_backtranslate_file(file_path):
    read_file = open(file_path, "r", encoding="utf-8")
    sentences = json.load(read_file)
    read_file.close()
    return paraphrase_sentences(sentences)


def main():
    paws_dataset = load_dataset("paws", "labeled_final")["train"]
    paws_paraphrases = process_paws_dataset(paws_dataset)

    sentences_for_backtranslate = "/kaggle/input/sentences-dataset/sentences_for_backtranslate.json"
    backtranslated_paraphrases = process_backtranslate_file(sentences_for_backtranslate)

    final_paraphrases = paws_paraphrases + backtranslated_paraphrases
    result_file = "/kaggle/working/combined_paraphrases.json"
    file = open(result_file, "w", encoding="utf-8")
    json.dump(final_paraphrases, file)
    file.close()


if __name__ == "__main__":
    main()

