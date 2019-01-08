#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N w2vAll
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
w2v_media=(All_OC All_OY All_OW All_PB All_PM All_PN)
# parallel --dry-run "\
#     python3 train.py --type $type --emb_type $emb_type --emb_path $emb_path/{2}.bin --gpu 0 --case {1} --dump_dir $type/$emb_type/{2}/{1} --emb_dim $emb_dim
#     python3 test.py --gpu 0 --load_dir $type/$emb_type/{2}/{1}
#     " ::: ${case[@]} ::: ${w2v_media[@]}

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All_OC.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/All_OC/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/All_OC/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All_OY.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/All_OY/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/All_OY/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All_OW.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/All_OW/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/All_OW/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All_PB.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/All_PB/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/All_PB/ni

# python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All_PM.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/All_PM/ni --emb_dim 200
# python3 test.py --gpu 0 --load_dir intra/Word2Vec/All_PM/ni

python3 train.py --type intra --emb_type Word2Vec --emb_path ../../data/embedding/Word2Vec/All_PN.bin --gpu 0 --case ni --dump_dir intra/Word2Vec/All_PN/ni --emb_dim 200
python3 test.py --gpu 0 --load_dir intra/Word2Vec/All_PN/ni