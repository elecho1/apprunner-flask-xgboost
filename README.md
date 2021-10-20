# Apprunner-Flask-XGBoost

Python3 × Flask × Docker x XGBoost推論のサンプル。

## セットアップ・動作確認

```
$ docker-compese up -d
```

[localhost:5000](http://localhost:5000/) にアクセスして「XGBoost prediction API with App Runner and flask.」と返ってくれば成功。

もしくは

```
$ curl http://localhost:5000/api/v1/predict -X POST -H 'Content-Type:application/json' -d '{"feature":[1, 1, 1, 1, 1, 1, 1, 1]}'

{"Content-Type":"application/json","pred":[1.7686777114868164],"success":true}
```

```
$ curl http://localhost:5000/api/v1/predict -X POST -H 'Content-Type:application/json' -d '{"feature":[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1, 1, 1, 1]]}'

{"Content-Type":"application/json","pred":[2.6185295581817627,1.7686777114868164],"success":true}

```

JSONが返ってくれば成功。

コンテナに入って何かしたい場合、

```
$ docker exec -it flask /bin/ash
```

で入れる。

# App Runnerへのデプロイ
(Todo)



## 参考リンク
* docker関連ファイルをサブディレクトリに配置する: https://qiita.com/mogya/items/73d2dae6c429926bf731
* Python(Flask) でサクッと 機械学習 API を作る: https://qiita.com/fam_taro/items/1464c42324f15d7b8223
* 