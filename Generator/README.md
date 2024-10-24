# Generator

Before LLM, the rag system of the previous encoder-decoder framework still needed to be trained, but after LLM came out, the threshold of the generator was greatly reduced because its parameters contained knowledge and had a certain ability to follow instructions. The generator of this project also focuses on how to use the existing model in combination with the retriever we trained in the previous step to implement the nanoRAG system.

## Requirements
same as Retriever.

## Testing NanoRAG

We implemented a generator based on the closed-source model OpenAI [GPT](https://openai.com/chatgpt/) and a generator based on the open-source model [LLaMA](https://github.com/meta-llama/llama3).

We can test the nanoRAG system using:

```
sh inference.sh --model gpt
```

You can define your own questions in `queries`, give it a try!

> You may need some API key or GPU resources to run the above command.

