import json
from bert_score import score
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


def load_paraphrases(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def compute_bleu(paraphrase_results, smoothie):
    bleu_scores = []
    for result in paraphrase_results:
        reference = [result["Text"].split()]
        candidate = result["Generated_Paraphrase"].split()
        bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_scores.append(bleu_score)
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    return average_bleu


def compute_rouge(paraphrase_results):
    rouge = Rouge()
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for result in paraphrase_results:
        if result["Generated_Paraphrase"].strip():
            rouge_score = rouge.get_scores(result["Generated_Paraphrase"], result["Reference_Paraphrase"])[0]
            rouge_1_scores.append(rouge_score['rouge-1']['f'])
            rouge_2_scores.append(rouge_score['rouge-2']['f'])
            rouge_l_scores.append(rouge_score['rouge-l']['f'])

    average_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
    average_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return average_rouge_1, average_rouge_2, average_rouge_l



def compute_bertscore(paraphrase_results):
    batch_size = 16
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for batch_results in tqdm(paraphrase_results, total=len(paraphrase_results) // batch_size, desc="BERTScore"):
        cands = [result["Generated_Paraphrase"] for result in batch_results]
        refs = [result["Reference_Paraphrase"] for result in batch_results]
        P, R, F1 = score(cands, refs, lang="en", batch_size=batch_size)
        precision_scores.extend(P.tolist())
        recall_scores.extend(R.tolist())
        f1_scores.extend(F1.tolist())
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)
    return average_precision, average_recall, average_f1


def compute_embedding_similarity(paraphrase_results, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    similarity_scores = []
    batch_size = 16
    for batch_results in tqdm(paraphrase_results, total=len(paraphrase_results) // batch_size, desc="Embedding Similarity"):
        texts = [result["Text"] for result in batch_results]
        paraphrases = [result["Generated_Paraphrase"] for result in batch_results]
        text_embeddings = model.encode(texts, convert_to_tensor=True)
        paraphrase_embeddings = model.encode(paraphrases, convert_to_tensor=True)
        similarities = util.cos_sim(text_embeddings, paraphrase_embeddings).diagonal()
        similarity_scores.extend(similarities.tolist())
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    return average_similarity


if __name__ == "__main__":
    paraphrase_results = load_paraphrases('/kaggle/input/results-score/paraphrase_results.json')
    smoothie = SmoothingFunction().method4
    average_bleu = compute_bleu(paraphrase_results, smoothie)
    average_rouge_1, average_rouge_2, average_rouge_l = compute_rouge(paraphrase_results)
    average_precision, average_recall, average_f1 = compute_bertscore(paraphrase_results)
    average_similarity = compute_embedding_similarity(paraphrase_results)

    print(f"Average BLEU score: {average_bleu}")
    print(f"Average ROUGE-1 F1 score: {average_rouge_1}")
    print(f"Average ROUGE-2 F1 score: {average_rouge_2}")
    print(f"Average ROUGE-L F1 score: {average_rouge_l}")
    print(f"Average BERTScore Precision: {average_precision}")
    print(f"Average BERTScore Recall: {average_recall}")
    print(f"Average BERTScore F1: {average_f1}")
    print(f"Average Embedding-Based Similarity: {average_similarity}")