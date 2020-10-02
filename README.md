# Keras reproducibility test

## 概要
- TensorFlow2.3 + kerasの再現性を確認するために書いたコードです．
- 交差検証を10回実行してすべて同じ結果になることを確認します．

## ファイル説明
- main.py : MNISTの訓練用の画像に対して交差検証(n=10)をして，10回分の学習の損失と精度(accuracy)を求める

- run_all.sh (Ubuntu用) or run_all.ps1(Win用): main.pyを10回繰り返して，交差検証の各回の分散を計算
    - 再現性を保った設定をした結果はwith_tf_option.txt，していない結果はwithout_tf_option.txtに保存

- eval.py : 保存したテキストファイルから分散の平均と処理時間を出力 
    - 注) 処理時間は結果を.txtに保存して次の結果を.txtに保存するまでの時間の合計なので，学習時間の正確な時間ではないです

- docker-compose.yml, env/Dockerfile: docker用のファイルです(Linux向け)
    - docker-composeが使える人は```docker-compose up```で交差検証の各回の分散の平均と処理時間を出力します
    - 生成したファイルがrootになることを防ぐために現在のユーザでコンテナを実行するようにしているので，実行前に```export UID```をしてください 

## 実行方法
- 1回だけ10交差検証を行う場合
    ```
    # 再現性を保たない場合
    > python main.py 0 
    
    # 再現性を保つ場合
    > python main.py 1

    ``` 
- 10回10交差検証をして分散と処理時間を出力する (再現性を保つ方法と保たない方法をそれぞれ10回実行して分散と処理時間を出力する)場合
    ```
    # Windows (Powershell)の場合
    > run_all.ps1

    # Ubuntu (Linux)の場合
    > run_all.sh
    ```
- Docker-composeをインストール済みの場合は10回の10交差検証を以下でできます
    ```
    > export UID
    > docker-compose up
    ```

## その他
- 不具合や改善点はIssuesもしくはtwitter(@10_2rugata)まで

## 履歴
- 2020/10/03 公開