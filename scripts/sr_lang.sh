lang=$1
port=$2
for fn in data/T1-test/${lang}_*; do
    f=`basename $fn | cut -d'.' -f1`
    echo $f
    mkdir -p gen/$f
    python surface/surface_realization.py --gen_dir gen/$f --test_file $fn --output_file output/$f.conllu --timeout 120 --port $port
done 
