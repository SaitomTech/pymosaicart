# python_dev_base

- [python_repository_simple](https://github.com/yamap55/python_repository_simple)を参考に、1からpython開発環境のベースを理解しながら作るレポジトリ

## 使い方

- `.env`ファイルを作成
- `xxxx`に作成予定のサービス名を追加
  - `WORKDIR` (Docker内の作業ディレクトリ)
    - `.devcontainer/Dockerfile/`
    - `docker-compose.yml`
    - `.devcontainer/devcontainer.json`: `workspaceFolder`
  - サービス, イメージ, コンテナ名
    - `docker-compose.yml`: `(service)`, `image`, `container_name`
    - `.devcontainer/devcontainer.json`: `service`
- `.devcontainer`の`name`に好きなプロジェクト名を入れる
- 左下の.devcontainerから`reopen in container`で使える
