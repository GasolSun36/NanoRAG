import os
import csv
import json
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, BertModel, BertTokenizer
import torch
from utils.tokenizers import SimpleTokenizer
import unicodedata
import time
import transformers
transformers.logging.set_verbosity_error()


import openai

def concatenate_documents(doc_list):
    result = ""
    for index, doc in enumerate(doc_list, start=1):
        result += f"{index}. {doc}\n"
    return result

def call_openai_api(question, docs, deployment_name):
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Here are some documents related to the question:\n" + concatenate_documents(docs) + 'Based on the documents, answer the following question:\n' + question}
        ],
        max_tokens=1000,
        temperature=0.1
    )
    result = response.choices[0].message.content
    return result

def call_llama_model(question, docs, pipeline):
    prompt = "Here are some documents related to the question:\n" + concatenate_documents(docs) + 'Based on the documents, answer the following question:\n' + question
    result = pipeline(prompt, max_length=1000, temperature=0.1)[0]['generated_text']
    return result

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_path", default="downloads/data/wikipedia_split/psgs_w100.tsv")
    parser.add_argument("--nq_test_file", default="downloads/data/retriever/qas/nq-test.csv")
    parser.add_argument("--encoding_batch_size", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--num_docs", type=int, default=21015324)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--wiki_passages_dump_path", default="wiki_passages.pkl")  # 新增参数
    parser.add_argument("--faiss_index_path", default="faiss_index.idx")  # 新增参数
    parser.add_argument("--model", choices=['gpt', 'llama'], default='gpt', help="Choose the language model to use")
    args = parser.parse_args()

    # 加载或保存 Wikipedia 文章
    if os.path.exists(args.wiki_passages_dump_path):
        # 从 dump 文件加载
        with open(args.wiki_passages_dump_path, 'rb') as f:
            wiki_passages = pickle.load(f)
        print(f"Loaded wikipedia passages from {args.wiki_passages_dump_path}")
    else:
        # 处理 TSV 文件并保存到 dump 文件
        id_col, text_col, title_col = 0, 1, 2
        wiki_passages = []
        with open(args.wikipedia_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in tqdm(reader, total=args.num_docs, desc="loading wikipedia passages..."):
                if row[id_col] == "id":
                    continue
                wiki_passages.append(row[text_col].strip('"'))
        # 保存到 dump 文件
        with open(args.wiki_passages_dump_path, 'wb') as f:
            pickle.dump(wiki_passages, f)
        print(f"Saved wikipedia passages to {args.wiki_passages_dump_path}")

    # 加载或保存 faiss 索引
    embedding_dimension = 768
    if os.path.exists(args.faiss_index_path):
        # 从文件加载已保存的索引
        index = faiss.read_index(args.faiss_index_path)
        print(f"Loaded faiss index from {args.faiss_index_path}")
    else:
        # 构建新的索引并保存
        index = faiss.IndexFlatIP(embedding_dimension)
        for idx in tqdm(range(args.num_shards), desc='building index from embedding...'):
            data = np.load(f"{args.embedding_dir}/wikipedia_shard_{idx}.npy")
            index.add(data)
        # 保存索引到文件
        faiss.write_index(index, args.faiss_index_path)
        print(f"Saved faiss index to {args.faiss_index_path}")

    # 加载查询编码器
    if args.pretrained_model_path == 'facebook/dpr-question_encoder-single-nq-base':
        query_encoder = DPRQuestionEncoder.from_pretrained(args.pretrained_model_path)
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        query_encoder = BertModel.from_pretrained(args.pretrained_model_path, add_pooling_layer=False)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()

    # 编码查询
    queries = ["Who is Donald Trump?"]
    query_embeddings = []
    for query in tqdm(queries, desc='encoding queries...'):
        with torch.no_grad():
            query_embedding = query_encoder(**tokenizer(query, max_length=256, truncation=True, padding='max_length', return_tensors='pt').to(device))
        if isinstance(query_encoder, DPRQuestionEncoder):
            query_embedding = query_embedding.pooler_output
        else:
            query_embedding = query_embedding.last_hidden_state[:, 0, :]
        query_embeddings.append(query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embeddings, axis=0)

    # 检索前 k 个文档
    print("searching index ", end=' ')
    start_time = time.time()
    top_k = 10
    _, I = index.search(query_embeddings, top_k)
    print(f"takes {time.time() - start_time} s")

    doc_lists = []
    I = I.flatten().tolist()
    for doc_id in I:
        doc = wiki_passages[doc_id]
        doc_lists.append(doc)

    test_question = "Who is Donald Trump?"

    if args.model == 'gpt':
        # 使用 OpenAI API
        # 设置 OpenAI API 参数，建议使用环境变量存储敏感信息
        openai.api_type = "azure"
        openai.api_base = os.environ.get("AZURE_OPENAI_API_BASE")
        openai.api_version = "2023-12-01-preview"
        openai.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        result = call_openai_api(test_question, doc_lists, deployment_name)
    elif args.model == 'llama':
        # 使用 Llama 模型
        model_id = "meta-llama/Meta-Llama-3.1-8B"
        pipeline = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        result = call_llama_model(test_question, doc_lists, pipeline)
    else:
        raise ValueError("Unsupported model type. Choose either 'gpt' or 'llama'.")

    print(result)
