#$ -cwd
#$ -l q_node=1
# 実行時間を指定（5分）
#$ -l h_rt=12:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N pred_intra_ga_0
# Module コマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn/7.1 gnuplot
# 自分のAnacondaとかjumanとか読ませるため
source /home/2/17M30683/.bash_profile
cd /gs/hs0/tga-cl/sango-m-ab/research2/PAS_by_torch/baseline/
# python3 train.py --type "intra" --epochs 10 --emb_type Random --emb_path "train_words.txt" --gpu 0 --case "ga" --dump_dir "intra/Random_Fix/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type Word2Vec --emb_path "../../data/embedding/Word2Vec/All.bin" --gpu 0 --case "ga" --dump_dir "intra/Word2Vec_Fix/All/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type Word2Vec --emb_path "../../data/embedding/Word2Vec/All_OC.bin" --gpu 0 --case "ga" --dump_dir "intra/Word2Vec_Fix/All_OC/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type Word2Vec --emb_path "../../data/embedding/Word2Vec/All_OY.bin" --gpu 0 --case "ga" --dump_dir "intra/Word2Vec_Fix/All_OY/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type Word2Vec --emb_path "../../data/embedding/Word2Vec/All_OW.bin" --gpu 0 --case "ga" --dump_dir "intra/Word2Vec_Fix/All_OW/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type Word2Vec --emb_path "../../data/embedding/Word2Vec/All_PB.bin" --gpu 0 --case "ga" --dump_dir "intra/Word2Vec_Fix/All_PB/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type Word2Vec --emb_path "../../data/embedding/Word2Vec/All_PM.bin" --gpu 0 --case "ga" --dump_dir "intra/Word2Vec_Fix/All_PM/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type Word2Vec --emb_path "../../data/embedding/Word2Vec/All_PN.bin" --gpu 0 --case "ga" --dump_dir "intra/Word2Vec_Fix/All_PN/ga" --emb_requires_grad_false
# python3 train.py --type "intra" --epochs 10 --emb_type FastText --emb_path "../../data/embedding/FastText/All.bin" --gpu 0 --case "ga" --dump_dir "intra/FastText_Fix/All/ga" --emb_requires_grad_false
python3 test.py --gpu 0 --load_dir "intra/Random/ga"
python3 test.py --gpu 0 --load_dir "intra/Random_Fix/ga"
python3 test.py --gpu 0 --load_dir "intra/Word2Vec/All/ga"
python3 test.py --gpu 0 --load_dir "intra/Word2Vec_Fix/All/ga"
python3 test.py --gpu 0 --load_dir "intra/Word2Vec/All_OC/ga"
python3 test.py --gpu 0 --load_dir "intra/Word2Vec_Fix/All_OC/ga"
