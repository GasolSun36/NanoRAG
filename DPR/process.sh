
$WIKI_DUMP_FILE_IN = 'data/wikipedia/enwiki-20210401-pages-articles-multistream.xml.bz2'
$WIKI_DUMP_FILE_OUT = 'data/wikipedia/enwiki-20210401-pages-articles-multistream.txt'

python3 -m wikiextractor.WikiExtractor  $WIKI_DUMP_FILE_IN --processes 8 -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT

sleep 3

python preprocess_wiki.py