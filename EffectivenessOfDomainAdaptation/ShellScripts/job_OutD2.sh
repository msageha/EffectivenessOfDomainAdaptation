#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N OutD2
# Module コマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn/7.1 gnuplot

# 自分のAnacondaとかjumanとか読ませるため
source /home/2/17M30683/.bash_profile
cd /gs/hs0/tga-cl/sango-m-ab/research2/PAS_by_torch/EffectivenessOfDomainAdaptation
#out PB
python3 train.py --emb_path '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt' --gpu 0 --case o --save --dump_dir './OutD/PB/o' --model Base --media 'OC', 'OY', 'OW', 'PM', 'PN'
python3 test.py --gpu 0 --load_dir './OutD/PB/o'

#out PM
python3 train.py --emb_path '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt' --gpu 0 --case o --save --dump_dir './OutD/PM/o' --model Base --media 'OC', 'OY', 'OW', 'PB', 'PN'
python3 test.py --gpu 0 --load_dir './OutD/PM/o'

#out PN
python3 train.py --emb_path '../../data/embedding/Word2VecWiki/entity_vector/entity_vector.model.txt' --gpu 0 --case o --save --dump_dir './OutD/PN/o' --model Base --media 'OC', 'OY', 'OW', 'PB', 'PM'
python3 test.py --gpu 0 --load_dir './OutD/PN/o'
