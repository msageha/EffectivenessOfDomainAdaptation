#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N BCCWJ_Modify
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

#w2v BCCWJ
python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All.bin --gpu 0 --case ga --dump_dir intra/Exophora_Modify/BCCWJ/ga --emb_dim 200
python3 test.py --gpu 0 --load_dir intra/Exophora_Modify/BCCWJ/ga

#w2v BCCWJ
python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All.bin --gpu 0 --case o --dump_dir intra/Exophora_Modify/BCCWJ/o --emb_dim 200
python3 test.py --gpu 0 --load_dir intra/Exophora_Modify/BCCWJ/o

#w2v BCCWJ
python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All.bin --gpu 0 --case ni --dump_dir intra/Exophora_Modify/BCCWJ/ni --emb_dim 200
python3 test.py --gpu 0 --load_dir intra/Exophora_Modify/BCCWJ/ni