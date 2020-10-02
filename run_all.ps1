Remove-Item ./results/with_tf_option.txt
Remove-Item ./results/without_tf_option.txt

for($i=0; $i -lt 10; $i++){
    python main.py 1
}

for($i=0; $i -lt 10; $i++){
    python main.py 0
}

python eval.py
# python main.py 1
