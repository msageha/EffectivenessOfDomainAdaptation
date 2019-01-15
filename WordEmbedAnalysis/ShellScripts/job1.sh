#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N w2v_Modify
# Module コマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn/7.1 gnuplot

# 自分のAnacondaとかjumanとか読ませるため
source /home/2/17M30683/.bash_profile
cd /gs/hs0/tga-cl/sango-m-ab/research2/PAS_by_torch/WordEmbedAnalysis

# configure
case=(ga)
type=intra
emb_type=Word2Vec
emb_path='../../data/embedding/Word2Vec'
emb_dim=200
w2v_media=(OC OY OW PB PM PN)

#w2v Wikipedia
python3 train.py --type intra --emb_type Word2VecWiki --emb_path ../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt --gpu 0 --case ga --dump_dir intra/Exophora_Modify/entity_vector/ga --emb_dim 200 --exo1_word '僕' --exo2_word 'おまえ' --exoX_word 'これ'
python3 test.py --gpu 0 --load_dir intra/Exophora_Modify/entity_vector/ga

#w2v Wikipedia
python3 train.py --type intra --emb_type Word2VecWiki --emb_path ../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt --gpu 0 --case o --dump_dir intra/Exophora_Modify/entity_vector/o --emb_dim 200 --exo1_word '僕' --exo2_word 'おまえ' --exoX_word 'これ'
python3 test.py --gpu 0 --load_dir intra/Exophora_Modify/entity_vector/o

#w2v Wikipedia
python3 train.py --type intra --emb_type Word2VecWiki --emb_path ../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt --gpu 0 --case ni --dump_dir intra/Exophora_Modify/entity_vector/ni --emb_dim 200 --exo1_word '僕' --exo2_word 'おまえ' --exoX_word 'これ'
python3 test.py --gpu 0 --load_dir intra/Exophora_Modify/entity_vector/ni