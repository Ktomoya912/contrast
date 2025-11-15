"""
コントラストゲームのAIモデル
PyTorchを用いた畳み込みニューラルネットワーク
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class ContrastCNN(nn.Module):
    """
    コントラストゲームの評価関数としてのCNN
    
    入力: 5x5のボード状態（複数チャンネル）
    出力: 盤面評価値（-1〜1）
    """
    
    def __init__(self, board_size: int = 5):
        super(ContrastCNN, self).__init__()
        
        self.board_size = board_size
        
        # 入力チャンネル数:
        # - プレイヤー1のコマ: 1ch
        # - プレイヤー2のコマ: 1ch
        # - 白タイル: 1ch
        # - 黒タイル: 1ch
        # - グレータイル: 1ch
        # - 現在の手番(1=P1, 0=P2): 1ch
        # - プレイヤー1のゴールまでの距離(正規化): 1ch
        # - プレイヤー2のゴールまでの距離(正規化): 1ch
        # - プレイヤー1の残り黒タイル数(全体): 1ch
        # - プレイヤー1の残りグレータイル数(全体): 1ch
        # - プレイヤー2の残り黒タイル数(全体): 1ch
        # - プレイヤー2の残りグレータイル数(全体): 1ch
        # 合計: 12ch
        in_channels = 12
        
        # 畳み込み層
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # バッチ正規化
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 全結合層
        self.fc1 = nn.Linear(256 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        順伝播
        
        Args:
            x: (batch_size, 12, board_size, board_size)
        
        Returns:
            評価値: (batch_size, 1) 範囲[-1, 1]
        """
        # 畳み込み層1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 畳み込み層2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 畳み込み層3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 平坦化
        x = x.view(x.size(0), -1)
        
        # 全結合層
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        # Tanhで[-1, 1]に正規化
        x = torch.tanh(x)
        
        return x
    
    def predict(self, board_state: np.ndarray) -> float:
        """
        単一の盤面評価
        
        Args:
            board_state: (12, board_size, board_size)
        
        Returns:
            評価値: float [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_state).unsqueeze(0)  # (1, 12, 5, 5)
            value = self.forward(x)
            return value.item()


class ValueNetwork:
    """
    価値ネットワークのラッパークラス
    学習と推論を管理
    """
    
    def __init__(self, board_size: int = 5, learning_rate: float = 0.001, use_cuda: bool = False):
        self.board_size = board_size
        
        # デバイスの設定（CUDA利用可能なら自動選択）
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"CUDA デバイスを使用: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            if use_cuda and not torch.cuda.is_available():
                print("警告: CUDAが利用できません。CPUを使用します。")
        
        self.model = ContrastCNN(board_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 学習履歴
        self.loss_history = []
    
    def train_step(self, states: List[np.ndarray], targets: List[float]) -> float:
        """
        1ステップの学習
        
        Args:
            states: 盤面状態のリスト
            targets: 目標価値のリスト
        
        Returns:
            損失値
        """
        self.model.train()
        
        # データをテンソルに変換してデバイスに転送
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        targets_tensor = torch.FloatTensor(targets).unsqueeze(1).to(self.device)
        
        # 勾配をリセット
        self.optimizer.zero_grad()
        
        # 順伝播
        predictions = self.model(states_tensor)
        
        # 損失計算
        loss = self.criterion(predictions, targets_tensor)
        
        # 逆伝播
        loss.backward()
        
        # パラメータ更新
        self.optimizer.step()
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        盤面を評価
        
        Args:
            state: 盤面状態 (5, board_size, board_size)
        
        Returns:
            評価値 [-1, 1]
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.model(x)
            return value.item()
    
    def save(self, path: str):
        """モデルを保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history
        }, path)
        print(f"モデルを保存しました: {path}")
    
    def load(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        print(f"モデルを読み込みました: {path}")


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
                    opponent = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
                    if opponent == Player.PLAYER1:
                        # プレイヤー1はy=0(上端)がゴール、距離=y
                        distance = y / (board_size - 1)
                    else:  # Player.PLAYER2
                        # プレイヤー2はy=4(下端)がゴール、距離=(4-y)
                        distance = (board_size - 1 - y) / (board_size - 1)
                    state[7, y, x] = distance
    
    # 残りタイル数（自分と相手）
    max_tiles = 5
    if current_player == Player.PLAYER1:
        my_black = game.tiles_remaining[Player.PLAYER1]['black'] / max_tiles
        my_gray = game.tiles_remaining[Player.PLAYER1]['gray'] / max_tiles
        opp_black = game.tiles_remaining[Player.PLAYER2]['black'] / max_tiles
        opp_gray = game.tiles_remaining[Player.PLAYER2]['gray'] / max_tiles
    else:
        my_black = game.tiles_remaining[Player.PLAYER2]['black'] / max_tiles
        my_gray = game.tiles_remaining[Player.PLAYER2]['gray'] / max_tiles
        opp_black = game.tiles_remaining[Player.PLAYER1]['black'] / max_tiles
        opp_gray = game.tiles_remaining[Player.PLAYER1]['gray'] / max_tiles
    
    state[8, :, :] = my_black
    state[9, :, :] = my_gray
    state[10, :, :] = opp_black
    state[11, :, :] = opp_gray
    
    return state


if __name__ == "__main__":
    # モデルのテスト
    print("ContrastCNN モデルのテスト")
    print("="*50)
    
    # モデルの初期化
    model = ValueNetwork(board_size=5, learning_rate=0.001)
    print(f"モデルのパラメータ数: {sum(p.numel() for p in model.model.parameters())}")
    
    # ダミーデータでテスト
    dummy_state = np.random.rand(12, 5, 5).astype(np.float32)
    value = model.evaluate(dummy_state)
    print(f"\nダミー盤面の評価値: {value:.4f}")
    
    # バッチ学習のテスト
    dummy_states = [np.random.rand(12, 5, 5).astype(np.float32) for _ in range(10)]
    dummy_targets = [np.random.rand() * 2 - 1 for _ in range(10)]
    
    loss = model.train_step(dummy_states, dummy_targets)
    print(f"学習損失: {loss:.6f}")
    
    print("\nモデルの構造:")
    print(model.model)
