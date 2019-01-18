#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N VOT
# Module コマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn/7.1 gnuplot

# 自分のAnacondaとかjumanとか読ませるため
source /home/2/17M30683/.bash_profile
cd /gs/hs0/tga-cl/sango-m-ab/research2/PAS_by_torch/EffectivenessOfDomainAdaptation
#ga
python3 voting.py --gpu 0 --load_FTdir './FT' --load_FAdir './FA/ga' --load_CPSdir './CPS/ga' --dump_dir './VOT'

#o
python3 voting.py --gpu 0 --load_FTdir './FT' --load_FAdir './FA/o' --load_CPSdir './CPS/o' --dump_dir './VOT'

#ni
python3 voting.py --gpu 0 --load_FTdir './FT' --load_FAdir './FA/ni' --load_CPSdir './CPS/ni' --dump_dir './VOT'
