python3 train.py --type "intra" --epochs 10 --emb_type Random --emb_path "train_words.txt" --gpu 0 --case "ga" --dump_path "intra/Random/ga" --media "OC"

usage: train.py [-h] --type {intra,inter} [--epochs MAX_EPOCH] --emb_type
                {Word2Vec,FastText,ELMo,Random} [--emb_path EMB_PATH]
                [--gpu GPU] [--batch BATCH_SIZE] --case {ga,o,ni}
                [--media {OC,OY,OW,PB,PM,PN} [{OC,OY,OW,PB,PM,PN} ...]]
                --dump_path DUMP_PATH

main function parser

optional arguments:
  -h, --help            show this help message and exit
  --type {intra,inter}  dataset: "intra" or "inter"
  --epochs MAX_EPOCH, -e MAX_EPOCH
                        max epoch
  --emb_type {Word2Vec,FastText,ELMo,Random}
                        word embedding type
  --emb_path EMB_PATH   word embedding path
  --gpu GPU, -g GPU     GPU ID for execution
  --batch BATCH_SIZE, -b BATCH_SIZE
                        mini batch size
  --case {ga,o,ni}, -c {ga,o,ni}
                        target "case" type
  --media {OC,OY,OW,PB,PM,PN} [{OC,OY,OW,PB,PM,PN} ...], -m {OC,OY,OW,PB,PM,PN} [{OC,OY,OW,PB,PM,PN} ...]
                        training media type
  --dump_path DUMP_PATH
                        model_dump_path