# nanoDPR

## Requirements
```bash
pip install -r requirements.txt
```

## Downloading Data for training
```python
python utils/download_data.py --resource data.wikipedia_split.psgs_w100
python utils/download_data.py --resource data.retriever.nq
python utils/download_data.py --resource data.retriever.qas.nq
```

## Training from scratch
First configure distributed setting and wandb setting:
```bash
accelerate config
wandb login
```
Then launch training with:
```bash
accelerate launch train_dpr.py
```
After training, we would get a trained **query encoder** and a **doc encoder**. 


## Evaluation
To evaluate the performance of retriever on the Natural Question dataset, firstly use **doc encoder** to encode all wikipedia passages:
```
accelerate launch doc2embedding.py \
    --pretrained_model_path model/path \
    --output_dir embedding/path

```
Then test DPR with:
```
python test_dpr.py --embedding_dir embedding/path --pretrained_model_path model/path
```

## Transfer to Recent Wiki Dump

The dump used in the above process is **wikipedia_split.psgs_w10**0, which is a dump of an older Wikipedia. A real RAG system requires a newer knowledge base, so we need to update Wikipedia to the latest version possible.

#### Download wikipedia dump

Use the [this link](https://dumps.wikimedia.org/backup-index.html) to download the latest dump, the suffix is ​​.xml.bz2, it is about 20G (We choose verson Sep. 2024).

#### Extracting, cleaning and Preprocess the wikipedia dump

First, we install the wikiextractor using:

```
git clone https://github.com/attardi/wikiextractor.git
```

Next, use wikiextractor to extracting and cleaning the downloaded wikipedia dump:

```
$WIKI_DUMP_FILE_IN = 'data/wikipedia/enwiki-20210401-pages-articles-multistream.xml.bz2'
$WIKI_DUMP_FILE_OUT = 'data/wikipedia/enwiki-20210401-pages-articles-multistream.txt'

python3 -m wikiextractor.WikiExtractor  $WIKI_DUMP_FILE_IN --processes 8 -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
```

Next, we need to preprocess the wikipedia dump:

```
python preprocess_wiki.py
```

#### Convert txt to tsv

Finally, we convert the processed Wikipedia dump into a tsv file in the same format as the old version for easy use later:

```
python dump_wiki.py
```

We have processed the latest Wikipedia dump and converted it into a tsv file. You can download it from this link: https://drive.google.com/file/d/1QQVY_Bdtdz3tHupaZyWqgVXGbENgSR4R/view?usp=sharing (the compressed file is about 6G, and the single file after decompression is about 20G)

> It is worth noting that since the NQ dataset used for training may have a different data distribution from the latest Wikipedia dump, the final retrieved text effect will be worse. However, since we divide the blocks by overlapping 10 tokens + 100 tokens per context, we can increase the number of topn, such as top-30.



Now, we can test our nanoRAG system.

## Testing NanoRAG

We can test the nanoRAG system using:

```
sh inference.sh
```

You can define your own questions in `queries`, give it a try!
