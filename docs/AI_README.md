# コントラストAI - TD学習による強化学習

PyTorchを用いたTD学習（Temporal Difference Learning）でコントラストゲームのAIを訓練します。

## 必要なパッケージ

```bash
pip3 install torch numpy matplotlib
# または
pip3 install -r requirements.txt
```

## AIの学習

### 基本的な学習

```bash
python3 train_ai.py --episodes 1000
```

### 詳細なオプション付き学習

```bash
python3 train_ai.py \
  --episodes 5000 \
  --learning-rate 0.001 \
  --discount 0.95 \
  --epsilon 0.2 \
  --lambda-td 0.7 \
  --save-interval 100 \
  --model-path my_ai.pth \
  --plot
```

### 既存モデルから学習を継続

```bash
python3 train_ai.py --episodes 1000 --load-model contrast_ai.pth
```

## AIと対戦

学習後、AIと対戦できます：

```bash
python3 play_with_ai.py
```

メニューから選択：
1. **AIと対戦** - 人間 vs AI
2. **AI同士の対戦を観戦** - AI vs AI を観戦

## アーキテクチャ

### ニューラルネットワーク構造

```
入力: 5×5×5 (ボード状態)
  ├─ チャンネル1: プレイヤー1のコマ
  ├─ チャンネル2: プレイヤー2のコマ
  ├─ チャンネル3: 白タイル
  ├─ チャンネル4: 黒タイル
  └─ チャンネル5: グレータイル

↓ Conv2D(5→32, 3×3) + BatchNorm + ReLU
↓ Conv2D(32→64, 3×3) + BatchNorm + ReLU
↓ Conv2D(64→128, 3×3) + BatchNorm + ReLU
↓ Flatten
↓ FC(128×5×5 → 256) + ReLU + Dropout(0.3)
↓ FC(256 → 128) + ReLU + Dropout(0.3)
↓ FC(128 → 1) + Tanh

出力: 盤面評価値 (-1〜1)
```

### TD学習アルゴリズム

1. **自己対戦**: AIが自分自身と対戦
2. **経験収集**: 各ゲームの状態と結果を記録
3. **価値更新**: TD誤差を用いて価値関数を更新
   ```
   V(s) ← V(s) + α[r + γV(s') - V(s)]
   ```
   - α: 学習率
   - γ: 割引率
   - r: 報酬（勝ち=1, 負け=-1, 引き分け=0）
4. **ε-greedy方策**: 探索と活用のバランス

## 学習パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--episodes` | 1000 | 学習エピソード数 |
| `--learning-rate` | 0.001 | 学習率（α） |
| `--discount` | 0.95 | 割引率（γ） |
| `--epsilon` | 0.2 | ε-greedyのε（探索率） |
| `--lambda-td` | 0.7 | TD(λ)のλ |
| `--save-interval` | 100 | モデル保存間隔 |
| `--model-path` | contrast_ai.pth | 保存先パス |

## ファイル構成

```
contrast/
├── ai_model.py           # CNNモデル定義
├── td_learning.py        # TD学習アルゴリズム
├── train_ai.py          # 学習実行スクリプト
├── play_with_ai.py      # AI対戦プログラム
├── requirements.txt     # 依存パッケージ
└── contrast_ai.pth      # 学習済みモデル（学習後に生成）
```

## 使用例

### 1. モデルの学習

```bash
# 1000エピソードで学習
python3 train_ai.py --episodes 1000 --plot
```

学習中の出力例：
```
Episode 10/1000 | Loss: 0.234567 | P1 Win: 52.00% | P2 Win: 48.00%
Episode 20/1000 | Loss: 0.198234 | P1 Win: 50.00% | P2 Win: 50.00%
...
```

### 2. AIと対戦

```bash
python3 play_with_ai.py
```

### 3. プログラムから利用

```python
from td_learning import TDLearner
from contrast_game import ContrastGame

# AIを読み込み
learner = TDLearner(board_size=5, epsilon=0.0)
learner.value_network.load("contrast_ai.pth")

# ゲームをプレイ
game = ContrastGame(board_size=5)
game.setup_initial_position()

# AIの手を取得
action = learner.select_action(game, use_epsilon=False)
```

## 学習のヒント

### 初期学習（探索重視）
```bash
python3 train_ai.py --episodes 2000 --epsilon 0.3 --learning-rate 0.001
```

### 精密化学習（活用重視）
```bash
python3 train_ai.py --episodes 3000 --epsilon 0.1 --learning-rate 0.0005 \
  --load-model contrast_ai.pth
```

### 長時間学習
```bash
# バックグラウンドで実行
nohup python3 train_ai.py --episodes 10000 --plot > training.log 2>&1 &
```

## パフォーマンス

- **学習速度**: 約10-20ゲーム/秒（CPU）
- **推奨エピソード数**: 1000-5000エピソード
- **モデルサイズ**: 約2-3MB

## トラブルシューティング

### PyTorchのインストールエラー

```bash
# CPU版のインストール
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

### メモリ不足

バッチサイズを小さくするか、エピソード数を分割して学習：

```bash
python3 train_ai.py --episodes 500
python3 train_ai.py --episodes 500 --load-model contrast_ai.pth
```

### 学習が収束しない

- 学習率を下げる: `--learning-rate 0.0005`
- εを下げる: `--epsilon 0.1`
- エピソード数を増やす: `--episodes 5000`

## 今後の拡張

- [ ] DQN（Deep Q-Network）の実装
- [ ] アルファベータ探索との組み合わせ
- [ ] モンテカルロ木探索（MCTS）
- [ ] マルチプロセス学習
- [ ] 対戦履歴の可視化
- [ ] ELOレーティングシステム
