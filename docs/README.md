# コントラスト - 2人対戦ボードゲーム

5×5の盤面で、タイルの色によって移動方向が変化する戦略的な2人対戦ボードゲームのPython実装です。

## 概要

コントラストは、タイルの配置戦略が重要な独特なルールを持つボードゲームです。通常は白いタイル上で縦横に移動しますが、プレイヤーは黒タイルとグレータイルを持っており、これらを戦略的に配置することで移動パターンを変化させることができます。

## 特徴

- **5×5の盤面**: コンパクトで戦略的なゲームプレイ
- **動的な移動パターン**: タイルの色によって移動可能な方向が変化
  - 白タイル(□): 縦横1マスに移動可能
  - 黒タイル(■): 斜め1マスのみ移動可能
  - グレータイル(▦): 8方向すべてに移動可能
- **戦略的なタイル配置**: 各プレイヤーが持つ特殊タイルを効果的に使う

## ゲームルール

### 基本ルール

1. **プレイヤー**: 2人で対戦
2. **ボード**: 5×5の盤面（初期状態はすべて白タイル）
3. **コマ**: 各プレイヤーは5つのコマを持つ
4. **移動**: コマは現在いるタイルの色によって移動可能な方向が決まる
   - **白タイル(□)**: 縦横1マス（上下左右）
   - **黒タイル(■)**: 斜め1マス（4方向）
   - **グレータイル(▦)**: 8方向すべて（縦横＋斜め）

### 特殊タイル

各プレイヤーは以下のタイルを持っています：
- **黒タイル × 3個**
- **グレータイル × 1個**

タイルは自分のターンに任意の空きマスに配置できますが、**一度配置すると移動できません**。

### 勝利条件

**相手の陣地（最初の列）に到達する**
- プレイヤー1（下側スタート）: 最上段（y=0）に到達
- プレイヤー2（上側スタート）: 最下段（y=4）に到達

## インストール

必要なもの:
- Python 3.7以上

```bash
# リポジトリをクローン（または直接ファイルをダウンロード）
cd contrast
```

## 使い方

### デモプログラムの実行

基本的なゲームの動作を確認:

```bash
python3 contrast_game.py
```

### インタラクティブゲームのプレイ

2人で対戦:

```bash
python3 interactive_game.py
```

ゲーム中の操作:
1. タイルを配置するか選択（y/n）
2. 配置する場合は、タイルの種類（black/gray）と位置を入力
3. 移動させるコマの座標を入力（X, Y）
4. 有効な移動先が表示される
5. 移動先の座標を入力（X, Y）

### ユニットテストの実行

```bash
python3 -m pytest test_contrast_game.py -v
# または
python3 test_contrast_game.py
```

## ファイル構成

```
contrast/
├── contrast_game.py         # メインゲームロジック
├── interactive_game.py      # インタラクティブなプレイ用
├── test_contrast_game.py    # ユニットテスト
└── README.md               # このファイル
```

## クラス構成

### `TileColor` (Enum)
タイルの色を定義
- `WHITE`: 白タイル（縦横移動）
- `BLACK`: 黒タイル（斜め移動）
- `GRAY`: グレータイル（全方向移動）

### `Player` (Enum)
プレイヤーを定義
- `PLAYER1`: プレイヤー1（下側スタート）
- `PLAYER2`: プレイヤー2（上側スタート）

### `Piece`
ゲームのコマ

**属性:**
- `owner`: コマの所有者

### `Board`
ゲームボード（5×5）

**属性:**
- `size`: ボードのサイズ（デフォルト5）
- `tiles`: 各マスのタイル色
- `pieces`: コマの配置

**メソッド:**
- `place_piece(x, y, piece)`: コマを配置
- `move_piece(from_x, from_y, to_x, to_y)`: コマを移動
- `get_piece(x, y)`: 指定位置のコマを取得
- `get_tile_color(x, y)`: 指定位置のタイル色を取得
- `set_tile_color(x, y, color)`: タイルの色を設定
- `display()`: ボードの状態を表示

### `ContrastGame`
ゲームのメインクラス

**属性:**
- `board`: ゲームボード
- `current_player`: 現在のプレイヤー
- `game_over`: ゲーム終了フラグ
- `winner`: 勝者
- `move_history`: 移動履歴
- `tiles_remaining`: 各プレイヤーの残りタイル数

**メソッド:**
- `setup_initial_position()`: 初期配置を設定
- `get_valid_moves(x, y)`: 有効な移動先を取得
- `place_tile(x, y, tile_color)`: タイルを配置
- `make_move(from_x, from_y, to_x, to_y, ...)`: コマを移動（タイル配置も可能）
- `display_game_state()`: ゲーム状態を表示

## ボードの表示記号

```
[□] - 白タイル（空）
[■] - 黒タイル（空）
[▦] - グレータイル（空）
[1□] - プレイヤー1のコマ（白タイル上）
[2■] - プレイヤー2のコマ（黒タイル上）
[1▦] - プレイヤー1のコマ（グレータイル上）
```

## 使用例

```python
from contrast_game import ContrastGame, Piece, Player, TileColor

# ゲームを作成（5×5ボード）
game = ContrastGame(board_size=5)
game.setup_initial_position()

# ゲーム状態を表示
game.display_game_state()

# 黒タイルを配置
game.place_tile(2, 3, TileColor.BLACK)

# 有効な移動先を取得
valid_moves = game.get_valid_moves(2, 4)
print(f"有効な移動先: {valid_moves}")

# コマを移動（タイル配置なし）
if valid_moves:
    to_x, to_y = valid_moves[0]
    game.make_move(2, 4, to_x, to_y)

# 移動後の状態を表示
game.display_game_state()

# タイル配置と移動を同時に行う
game.make_move(
    from_x=1, from_y=0, 
    to_x=1, to_y=1,
    place_tile=True,
    tile_x=1, tile_y=2,
    tile_color=TileColor.GRAY
)
```

## カスタマイズ

### ボードサイズの変更

```python
# カスタムサイズのボードでプレイ（非推奨：バランスが崩れる可能性）
game = ContrastGame(board_size=7)
```

### カスタム初期配置

```python
game = ContrastGame(board_size=5)

# 手動で初期配置を設定
for x in range(3):
    piece = Piece(owner=Player.PLAYER1)
    game.board.place_piece(x, 4, piece)

# ... 他のコマを配置 ...
```

## 拡張アイデア

このプログラムを拡張するためのアイデア:

1. **AIプレイヤーの実装**
   - ミニマックスアルゴリズム
   - アルファベータ枝刈り
   - モンテカルロ木探索
   - タイル配置のヒューリスティック評価

2. **GUI版の作成**
   - Pygame（ドラッグ&ドロップでコマを移動）
   - Tkinter（シンプルなGUI）
   - Web版（Flask/Django + JavaScript）

3. **ネットワーク対戦**
   - ソケット通信
   - オンラインマルチプレイヤー
   - マッチメイキングシステム

4. **ゲーム分析機能**
   - 棋譜の保存と再生
   - 最善手の提案
   - ゲーム統計の記録

5. **ゲームのバリエーション**
   - 異なるタイル構成（黒×4、グレー×2など）
   - より大きな盤面（7×7など）
   - 追加の特殊タイル（ワープ、壁など）
   - 3人以上での対戦モード

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告、機能要望、プルリクエストを歓迎します。

## 作者

コントラストゲームのPython実装

## 参考

オリジナルのゲーム「コントラスト」は029PRODUCTによって制作されました。
