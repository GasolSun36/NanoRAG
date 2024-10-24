import os

def process_text_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    words = text.split()
    
    segment_length = 100
    overlap = 10
    start = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write("id\ttext\ttitle\n")
        
        id_counter = 1
        
        while start < len(words):
            end = start + segment_length
            segment = words[start:end]
            
            out_file.write(f"{id_counter}\t{' '.join(segment)}\tNone\n")
            id_counter += 1
            
            start += (segment_length - overlap)


input_file = 'enwiki-20240901-pages-articles-multistream_preprocessed.txt'
output_file = '20240901_wiki_dump.tsv'
process_text_file(input_file, output_file)
