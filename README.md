# Deep Learning Tuningathon用ベースコード

## 環境準備

```
pip install -r requirements.txt
```

## 実行方法
下記コマンドを実施してください

```
python cifar10.py <path to model>

# ex)
python cifar10.py models.ex01
```

デフォルトで幾つかのオプションを持っています

```
python cifar10.py models.ex01 --batch_size 32 --num_epoch 10 --data_augmentation
```

## 使い方

### モデルの追加

*models/* にpythonのファイルと下記フォーマットの関数を追加して下さい。

```python
def create_network(input_shape, output_shape, some_args):
    model = Sequential()
        :
    return model
```

その後追加したファイル名を指定して下さい。

```
# ex) newmodel.pyというファイルを追加した場合
python cifar10.py models/newmodel
```


