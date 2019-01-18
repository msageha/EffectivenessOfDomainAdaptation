#$ -cwd
#$ -l s_gpu=1
# 実行時間を指定（5分）
#$ -l h_rt=24:00:00
# 名前（hill_climbing.e[ジョブ番号？]，hill_climbing.o[ジョブ番号？]というそれぞれエラー出力，標準出力ファイルが生成される．ただしこれの内容は信用できない）
#$ -N FT
# Module コマンドの初期化
. /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn/7.1 gnuplot

# 自分のAnacondaとかjumanとか読ませるため
source /home/2/17M30683/.bash_profile
cd /gs/hs0/tga-cl/sango-m-ab/research2/PAS_by_torch/EffectivenessOfDomainAdaptation
#ga
# python3 fine_tuning.py --gpu 0 --load_dir './Base/ga' --model Base --dump_dir './FT'
python3 test.py --gpu 0 --load_dir './FT/OC/ga'
python3 test.py --gpu 0 --load_dir './FT/OY/ga'
python3 test.py --gpu 0 --load_dir './FT/OW/ga'
python3 test.py --gpu 0 --load_dir './FT/PB/ga'
python3 test.py --gpu 0 --load_dir './FT/PM/ga'
python3 test.py --gpu 0 --load_dir './FT/PN/ga'

#o
# python3 fine_tuning.py --gpu 0 --load_dir './Base/o' --model Base --dump_dir './FT'
python3 test.py --gpu 0 --load_dir './FT/OC/o'
python3 test.py --gpu 0 --load_dir './FT/OY/o'
python3 test.py --gpu 0 --load_dir './FT/OW/o'
python3 test.py --gpu 0 --load_dir './FT/PB/o'
python3 test.py --gpu 0 --load_dir './FT/PM/o'
python3 test.py --gpu 0 --load_dir './FT/PN/o'

#ni
# python3 fine_tuning.py --gpu 0 --load_dir './Base/ni' --model Base --dump_dir './FT'
python3 test.py --gpu 0 --load_dir './FT/OC/ni'
python3 test.py --gpu 0 --load_dir './FT/OY/ni'
python3 test.py --gpu 0 --load_dir './FT/OW/ni'
python3 test.py --gpu 0 --load_dir './FT/PB/ni'
python3 test.py --gpu 0 --load_dir './FT/PM/ni'
python3 test.py --gpu 0 --load_dir './FT/PN/ni'
