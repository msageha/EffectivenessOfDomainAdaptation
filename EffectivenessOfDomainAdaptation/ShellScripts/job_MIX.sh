#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N MIX
# Module コマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn/7.1 gnuplot

# 自分のAnacondaとかjumanとか読ませるため
source /home/2/17M30683/.bash_profile
cd /gs/hs0/tga-cl/sango-m-ab/research2/PAS_by_torch/EffectivenessOfDomainAdaptation
#ga
python3 train.py --emb_path '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt' --gpu 0 --case ga --save --dump_dir './MIX/ga' --model MIX
python3 test.py --gpu 0 --load_dir './MIX/ga'

#o
python3 train.py --emb_path '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt' --gpu 0 --case o --save --dump_dir './MIX/o' --model MIX
python3 test.py --gpu 0 --load_dir './MIX/o'

#ni
python3 train.py --emb_path '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt' --gpu 0 --case ni --save --dump_dir './MIX/ni' --model MIX
python3 test.py --gpu 0 --load_dir './MIX/ni'
