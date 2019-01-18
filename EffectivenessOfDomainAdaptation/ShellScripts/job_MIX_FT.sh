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
# python3 fine_tuning.py --gpu 0 --load_dir './MIX_Base/ga' --model MIX --dump_dir './MIX'
python3 test.py --gpu 0 --load_dir './MIX/OC/ga'
python3 test.py --gpu 0 --load_dir './MIX/OY/ga'
python3 test.py --gpu 0 --load_dir './MIX/OW/ga'
python3 test.py --gpu 0 --load_dir './MIX/PB/ga'
python3 test.py --gpu 0 --load_dir './MIX/PM/ga'
python3 test.py --gpu 0 --load_dir './MIX/PN/ga'

#o
# python3 fine_tuning.py --gpu 0 --load_dir './MIX_Base/o' --model MIX --dump_dir './MIX'
python3 test.py --gpu 0 --load_dir './MIX/OC/o'
python3 test.py --gpu 0 --load_dir './MIX/OY/o'
python3 test.py --gpu 0 --load_dir './MIX/OW/o'
python3 test.py --gpu 0 --load_dir './MIX/PB/o'
python3 test.py --gpu 0 --load_dir './MIX/PM/o'
python3 test.py --gpu 0 --load_dir './MIX/PN/o'

#ni
# python3 fine_tuning.py --gpu 0 --load_dir './MIX_Base/ni' --model MIX --dump_dir './MIX'
python3 test.py --gpu 0 --load_dir './MIX/OC/ni'
python3 test.py --gpu 0 --load_dir './MIX/OY/ni'
python3 test.py --gpu 0 --load_dir './MIX/OW/ni'
python3 test.py --gpu 0 --load_dir './MIX/PB/ni'
python3 test.py --gpu 0 --load_dir './MIX/PM/ni'
python3 test.py --gpu 0 --load_dir './MIX/PN/ni'
