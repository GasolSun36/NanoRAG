# NanoRAG

RAG (Retrieval-augmented Generation) is a widely used technique in the field of NLP (Natural Language Processing) and is a highly popular topic in both academia and industry. However, current RAG systems (such as [Langchain](https://github.com/langchain-ai/langchain)) involve significant engineering effort and a large number of parameters, which can be challenging for beginners to learn. Therefore, the original intent of this project is to utilize the simplest and most classic techniques to build a RAG system from scratch, providing code implementations for key steps only.


## Key Components for RAG

The two most critical components of rag are the **Retriever** and the **Generator**. The retriever is responsible for returning the top n most relevant documents in the database based on the input question and handing them over to the generator. The generator generates the corresponding answer based on the question and the top n documents. For now, we will only focus on **plain text**.

### Retriever

There are many papers related to the retrieval. The retrieval of this project uses the most basic retrieval DPR (Dense Passage Retrieval). This project refers to [this](https://github.com/Hannibal046/nanoDPR).

### Generator

With the emergence of LLM, decoder-only generators have become popular. This project implements generators based on OpenAI's [GPT](https://openai.com/chatgpt/) (closed source) and [LLaMA](https://github.com/meta-llama/llama3) (open source).


## Todolist

<input type="checkbox" checked> 
<label>DPR实现</label><br>

<input type="checkbox" disabled> 
<label>DPR的inference出doc原文</label>

<input type="checkbox" disabled> 
<label>Generator实现</label>

<input type="checkbox" disabled> 
<label>几个benchmark上测试</label>

<input type="checkbox" disabled> 
<label>database拓展成最新的wikidump</label>

<input type="checkbox" disabled> 
<label>支持图片输入</label>