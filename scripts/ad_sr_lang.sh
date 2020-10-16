lang=$1
port=$2
for fn in data/2020_additional/T1-test/${lang}_*; do
    f=`basename $fn | cut -d'.' -f1`
    echo $f
    mkdir -p gen/ad/$f
    python surface/surface_realization.py --gen_dir gen/ad/$f --test_file $fn --output_file output/ad/$f.conllu --timeout 120 --port $port
done 
