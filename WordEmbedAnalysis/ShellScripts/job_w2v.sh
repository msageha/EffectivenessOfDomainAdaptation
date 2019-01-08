#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N w2v
# Module コマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn/7.1 gnuplot

# 自分のAnacondaとかjumanとか読ませるため
source /home/2/17M30683/.bash_profile
cd /gs/hs0/tga-cl/sango-m-ab/research2/PAS_by_torch/WordEmbedAnalysis

# configure
case=(ni)
type=intra
emb_type=Word2Vec
emb_path='../../data/embedding/Word2Vec'
emb_dim=200
w2v_media=(OC OY OW PB PM PN)
# parallel --dry-run "\
#     python3 train.py --type $type --emb_type $emb_type --emb_path $emb_path/{2}.bin --gpu 0 --case {1} --dump_dir $type/$emb_type/{2}/{1} --emb_dim $emb_dim
#     python3 test.py --gpu 0 --load_dir $type/$emb_type/{2}/{1}
#     " ::: ${case[@]} ::: ${w2v_media[@]}

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/OC.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/OC/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/OC/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/OY.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/OY/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/OY/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/OW.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/OW/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/OW/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/PB.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/PB/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/PB/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/PM.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/PM/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/PM/ni

python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/PN.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/PN/ni --emb_dim 200
python3 test.py --gpu 0 --load_dir intra/Word2Vec/PN/ni