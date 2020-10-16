lang=$1
port=$2
export FLASK_ENV=
python surface/grammar.py --train_files data/T1-train/${lang}_* --test_files data/T1-test/${lang}_* --model_file models/$lang.bin --port $port
