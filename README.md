# python3-flask-docker-sample

Python3 × Flask × Dockerのサンプル。

## セットアップ

```
$ docker-compese build
```

## 動作確認

```
$ docker-compese up -d
```

[localhost:5000](http://localhost:5000/) にアクセスして「Hello World!」と返ってくれば成功。

もしくは

```
$ curl http://localhost:5000/api/v1/hello

{
  "message": "Hello World!"
}
```

JSONが返ってくれば成功。

コンテナに入って何かしたい場合、

```
$ docker exec -it flask /bin/ash
```

で入れる。
