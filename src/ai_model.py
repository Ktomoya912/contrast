"""
コントラストゲームのAIモデル (Actor-Critic)
PyTorchを用いた畳み込みニューラルネットワーク
PolicyとValueを同時に学習
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ロガー設定
logger = logging.getLogger(__name__)


class ContrastCNN(nn.Module):
    """
    コントラストゲームのActor-Criticネットワーク（移動とタイル分離版）

    入力: 5x5のボード状態（12チャンネル）
    出力:
        - Move Policy: 駒の移動確率分布 (25 * 8 = 200次元)
        - Tile Policy: タイル配置確率分布 (31次元: なし1 + 黒15 + グレー15)
        - Value: 盤面評価値（-1〜1）
    """

    def __init__(self, board_size: int = 5):
        super(ContrastCNN, self).__init__()

        self.board_size = board_size
        # 移動: 25箇所(from) × 8方向(dir) = 200通り
        self.max_move_actions = board_size * board_size * 8
        # タイル配置: なし(1) + 黒タイル(15) + グレータイル(15) = 31通り
        self.max_tile_actions = 31

        # 入力チャンネル数: 12ch
        in_channels = 12

        # 共有畳み込み層（特徴抽出）
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # 共有全結合層
        self.fc_shared = nn.Linear(256 * board_size * board_size, 512)

        # Move Policy Head（駒の移動）
        self.move_fc1 = nn.Linear(512, 256)
        self.move_fc2 = nn.Linear(256, self.max_move_actions)

        # Tile Policy Head（タイル配置）
        self.tile_fc1 = nn.Linear(512, 128)
        self.tile_fc2 = nn.Linear(128, self.max_tile_actions)

        # Value Head（盤面評価）
        self.value_fc1 = nn.Linear(512, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, move_mask=None, tile_mask=None):
        """
        順伝播

        Args:
            x: (batch_size, 12, board_size, board_size)
            move_mask: (batch_size, max_move_actions) 移動の合法手マスク（オプション）
            tile_mask: (batch_size, max_tile_actions) タイル配置の合法手マスク（オプション）

        Returns:
            move_logits: (batch_size, max_move_actions) 移動ロジット
            tile_logits: (batch_size, max_tile_actions) タイル配置ロジット
            value: (batch_size, 1) 盤面評価値 [-1, 1]
        """
        # 共有畳み込み層
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 平坦化
        x = x.view(x.size(0), -1)

        # 共有全結合層
        shared = self.fc_shared(x)
        shared = F.relu(shared)
        shared = self.dropout(shared)

        # Move Policy Head（駒の移動）
        move = self.move_fc1(shared)
        move = F.relu(move)
        move = self.dropout(move)
        move_logits = self.move_fc2(move)

        # 合法手マスクを適用（不正な手に大きな負の値）
        if move_mask is not None:
            move_logits = move_logits.masked_fill(~move_mask, -1e9)

        # Tile Policy Head（タイル配置）
        tile = self.tile_fc1(shared)
        tile = F.relu(tile)
        tile = self.dropout(tile)
        tile_logits = self.tile_fc2(tile)

        # 合法手マスクを適用
        if tile_mask is not None:
            tile_logits = tile_logits.masked_fill(~tile_mask, -1e9)

        # Value Head
        value = self.value_fc1(shared)
        value = F.relu(value)
        value = self.dropout(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)  # [-1, 1]に正規化

        return move_logits, tile_logits, value


class ActorCriticNetwork:
    """
    Actor-Criticネットワークのラッパークラス
    PolicyとValueの両方を学習・推論
    """

    def __init__(
        self,
        board_size: int = 5,
        max_actions: int = 500,  # 互換性のため残すが使用しない
        learning_rate: float = 0.0003,
        use_cuda: bool = False,
    ):
        self.board_size = board_size
        # 新しい行動空間
        self.max_move_actions = board_size * board_size * 8  # 200
        self.max_tile_actions = 31  # なし1 + 黒15 + グレー15

        # デバイスの設定（CUDA利用可能なら自動選択）
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"CUDA デバイスを使用: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            if use_cuda and not torch.cuda.is_available():
                logger.warning("CUDAが利用できません。CPUを使用します。")

        self.model = ContrastCNN(board_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        logger.info(
            f"Move actions: {self.max_move_actions}, Tile actions: {self.max_tile_actions}"
        )

        # 学習履歴
        self.move_loss_history = []
        self.tile_loss_history = []
        self.value_loss_history = []
        self.total_loss_history = []

    def train_step(
        self,
        states: List[np.ndarray],
        move_indices: List[int],
        tile_indices: List[int],
        move_masks: List[np.ndarray],
        tile_masks: List[np.ndarray],
        advantages: List[float],
        value_targets: List[float],
    ) -> Tuple[float, float, float, float]:
        """
        1ステップのActor-Critic学習（移動とタイル分離版）

        Args:
            states: 盤面状態のリスト
            move_indices: 実際に取った移動のインデックス
            tile_indices: 実際に取ったタイル配置のインデックス
            move_masks: 移動の合法手マスク
            tile_masks: タイル配置の合法手マスク
            advantages: Advantage値（Q - V）
            value_targets: Value目標値

        Returns:
            (move_loss, tile_loss, value_loss, total_loss)
        """
        self.model.train()

        # データをテンソルに変換してデバイスに転送
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        move_indices_tensor = torch.LongTensor(move_indices).to(self.device)
        tile_indices_tensor = torch.LongTensor(tile_indices).to(self.device)
        move_masks_tensor = torch.BoolTensor(np.array(move_masks)).to(self.device)
        tile_masks_tensor = torch.BoolTensor(np.array(tile_masks)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        value_targets_tensor = (
            torch.FloatTensor(value_targets).unsqueeze(1).to(self.device)
        )

        # 勾配をリセット
        self.optimizer.zero_grad()

        # 順伝播
        move_logits, tile_logits, value_preds = self.model(
            states_tensor, move_masks_tensor, tile_masks_tensor
        )

        # Move Policy Loss
        move_log_probs = F.log_softmax(move_logits, dim=-1)
        move_action_log_probs = move_log_probs.gather(
            1, move_indices_tensor.unsqueeze(1)
        ).squeeze(1)
        move_policy_loss = -(move_action_log_probs * advantages_tensor).mean()

        # Tile Policy Loss
        tile_log_probs = F.log_softmax(tile_logits, dim=-1)
        tile_action_log_probs = tile_log_probs.gather(
            1, tile_indices_tensor.unsqueeze(1)
        ).squeeze(1)
        tile_policy_loss = -(tile_action_log_probs * advantages_tensor).mean()

        # エントロピーボーナス（探索促進）
        move_probs = F.softmax(move_logits, dim=-1)
        move_entropy = -(move_probs * move_log_probs).sum(dim=-1).mean()
        tile_probs = F.softmax(tile_logits, dim=-1)
        tile_entropy = -(tile_probs * tile_log_probs).sum(dim=-1).mean()

        # Value Loss（MSE）
        value_loss = F.mse_loss(value_preds, value_targets_tensor)

        # 総合損失
        total_loss = (
            move_policy_loss
            + tile_policy_loss
            + 0.5 * value_loss
            - 0.01 * (move_entropy + tile_entropy)
        )

        # 逆伝播
        total_loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        # パラメータ更新
        self.optimizer.step()

        # 履歴に追加
        move_loss_val = move_policy_loss.item()
        tile_loss_val = tile_policy_loss.item()
        value_loss_val = value_loss.item()
        total_loss_val = total_loss.item()

        self.move_loss_history.append(move_loss_val)
        self.tile_loss_history.append(tile_loss_val)
        self.value_loss_history.append(value_loss_val)
        self.total_loss_history.append(total_loss_val)

        return move_loss_val, tile_loss_val, value_loss_val, total_loss_val

    def evaluate(
        self,
        state: np.ndarray,
        move_mask: np.ndarray = None,
        tile_mask: np.ndarray = None,
    ):
        """
        盤面を評価して移動Policy、タイルPolicy、Valueを返す

        Args:
            state: 盤面状態 (12, board_size, board_size)
            move_mask: 移動の合法手マスク (max_move_actions,)
            tile_mask: タイル配置の合法手マスク (max_tile_actions,)

        Returns:
            move_probs: 移動確率分布 (max_move_actions,)
            tile_probs: タイル配置確率分布 (max_tile_actions,)
            value: 評価値 [-1, 1]
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            m_mask = (
                torch.BoolTensor(move_mask).unsqueeze(0).to(self.device)
                if move_mask is not None
                else None
            )
            t_mask = (
                torch.BoolTensor(tile_mask).unsqueeze(0).to(self.device)
                if tile_mask is not None
                else None
            )

            move_logits, tile_logits, value = self.model(x, m_mask, t_mask)
            move_probs = F.softmax(move_logits, dim=-1)
            tile_probs = F.softmax(tile_logits, dim=-1)

            return (
                move_probs.squeeze(0).cpu().numpy(),
                tile_probs.squeeze(0).cpu().numpy(),
                value.item(),
            )

    def save(self, path: str):
        """モデルを保存"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "move_loss_history": self.move_loss_history,
                "tile_loss_history": self.tile_loss_history,
                "value_loss_history": self.value_loss_history,
                "total_loss_history": self.total_loss_history,
            },
            path,
        )
        logger.info(f"モデルを保存しました: {path}")

    def load(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.move_loss_history = checkpoint.get("move_loss_history", [])
        self.tile_loss_history = checkpoint.get("tile_loss_history", [])
        self.value_loss_history = checkpoint.get("value_loss_history", [])
        self.total_loss_history = checkpoint.get("total_loss_history", [])
        logger.info(f"モデルを読み込みました: {path}")


def board_to_tensor(game) -> np.ndarray:
    """
    ゲーム状態をニューラルネットワークの入力形式に変換
    常に現在のプレイヤー視点で正規化（自分=ch0, 相手=ch1）

    Args:
        game: ContrastGame オブジェクト

    Returns:
        (12, board_size, board_size) の numpy 配列
    """
    from contrast_game import Player, TileColor

    board_size = game.board.size
    state = np.zeros((12, board_size, board_size), dtype=np.float32)

    current_player = game.current_player

    for y in range(board_size):
        for x in range(board_size):
            # コマの配置（常に現在プレイヤー視点）
            piece = game.board.get_piece(x, y)
            if piece:
                if piece.owner == current_player:
                    state[0, y, x] = 1.0  # 自分のコマ
                else:
                    state[1, y, x] = 1.0  # 相手のコマ

            # タイルの色
            tile_color = game.board.get_tile_color(x, y)
            if tile_color == TileColor.WHITE:
                state[2, y, x] = 1.0
            elif tile_color == TileColor.BLACK:
                state[3, y, x] = 1.0
            else:  # GRAY
                state[4, y, x] = 1.0

    # 手番情報は不要（常に現在プレイヤー視点なので常に1.0）
    state[5, :, :] = 1.0

    # ゴールまでの距離（自分と相手、現在プレイヤー視点）
    for y in range(board_size):
        for x in range(board_size):
            piece = game.board.get_piece(x, y)
            if piece:
                # 自分のコマのゴール距離
                if piece.owner == current_player:
                    if current_player == Player.PLAYER1:
                        # プレイヤー1はy=0(上端)がゴール、距離=y
                        distance = y / (board_size - 1)
                    else:  # Player.PLAYER2
                        # プレイヤー2はy=4(下端)がゴール、距離=(4-y)
                        distance = (board_size - 1 - y) / (board_size - 1)
                    state[6, y, x] = distance
                # 相手のコマのゴール距離
                else:
                    opponent = (
                        Player.PLAYER2
                        if current_player == Player.PLAYER1
                        else Player.PLAYER1
                    )
                    if opponent == Player.PLAYER1:
                        # プレイヤー1はy=0(上端)がゴール、距離=y
                        distance = y / (board_size - 1)
                    else:  # Player.PLAYER2
                        # プレイヤー2はy=4(下端)がゴール、距離=(4-y)
                        distance = (board_size - 1 - y) / (board_size - 1)
                    state[7, y, x] = distance

    # 残りタイル数（自分と相手）
    # 黒タイルの最大数: 3、グレータイルの最大数: 1
    max_black_tiles = 3
    max_gray_tiles = 1

    if current_player == Player.PLAYER1:
        my_black = game.tiles_remaining[Player.PLAYER1]["black"] / max_black_tiles
        my_gray = game.tiles_remaining[Player.PLAYER1]["gray"] / max_gray_tiles
        opp_black = game.tiles_remaining[Player.PLAYER2]["black"] / max_black_tiles
        opp_gray = game.tiles_remaining[Player.PLAYER2]["gray"] / max_gray_tiles
    else:
        my_black = game.tiles_remaining[Player.PLAYER2]["black"] / max_black_tiles
        my_gray = game.tiles_remaining[Player.PLAYER2]["gray"] / max_gray_tiles
        opp_black = game.tiles_remaining[Player.PLAYER1]["black"] / max_black_tiles
        opp_gray = game.tiles_remaining[Player.PLAYER1]["gray"] / max_gray_tiles

    state[8, :, :] = my_black
    state[9, :, :] = my_gray
    state[10, :, :] = opp_black
    state[11, :, :] = opp_gray

    return state


if __name__ == "__main__":
    # モデルのテスト
    logger.info("Actor-Critic モデルのテスト")
    logger.info("=" * 50)

    # モデルの初期化
    model = ActorCriticNetwork(board_size=5, learning_rate=0.0003)
    logger.info(
        f"モデルのパラメータ数: {sum(p.numel() for p in model.model.parameters())}"
    )

    # ダミーデータでテスト
    dummy_state = np.random.rand(12, 5, 5).astype(np.float32)
    dummy_move_mask = np.random.rand(200) > 0.9  # ランダムに10%を合法手とする
    dummy_tile_mask = np.random.rand(31) > 0.5  # ランダムに50%を合法とする

    move_probs, tile_probs, value = model.evaluate(
        dummy_state, dummy_move_mask, dummy_tile_mask
    )
    logger.info(f"ダミー盤面の評価値: {value:.4f}")
    logger.info(f"合法な移動の数: {dummy_move_mask.sum()}")
    logger.info(f"合法なタイル配置の数: {dummy_tile_mask.sum()}")
    logger.info(f"最大確率の移動: {move_probs.argmax()}, 確率: {move_probs.max():.4f}")
    logger.info(
        f"最大確率のタイル配置: {tile_probs.argmax()}, 確率: {tile_probs.max():.4f}"
    )

    # バッチ学習のテスト
    batch_size = 10
    dummy_states = [
        np.random.rand(12, 5, 5).astype(np.float32) for _ in range(batch_size)
    ]
    dummy_move_actions = [np.random.randint(0, 200) for _ in range(batch_size)]
    dummy_tile_actions = [np.random.randint(0, 31) for _ in range(batch_size)]
    dummy_move_masks = [np.random.rand(200) > 0.9 for _ in range(batch_size)]
    dummy_tile_masks = [np.random.rand(31) > 0.5 for _ in range(batch_size)]
    dummy_advantages = [np.random.rand() * 2 - 1 for _ in range(batch_size)]
    dummy_value_targets = [np.random.rand() * 2 - 1 for _ in range(batch_size)]

    m_loss, t_loss, v_loss, total_loss = model.train_step(
        dummy_states,
        dummy_move_actions,
        dummy_tile_actions,
        dummy_move_masks,
        dummy_tile_masks,
        dummy_advantages,
        dummy_value_targets,
    )
    logger.info("学習損失:")
    logger.info(f"  Move Loss: {m_loss:.6f}")
    logger.info(f"  Tile Loss: {t_loss:.6f}")
    logger.info(f"  Value Loss: {v_loss:.6f}")
    logger.info(f"  Total Loss: {total_loss:.6f}")

    logger.info("モデルの構造:")
    logger.info(str(model.model))
