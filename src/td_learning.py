"""
TD学習（Temporal Difference Learning）による強化学習
自己対戦を通じてコントラストゲームのAIを訓練
"""

import numpy as np
import copy
from typing import List, Tuple, Optional
from collections import deque
import random

from contrast_game import ContrastGame, Player, TileColor
from ai_model import ValueNetwork, board_to_tensor


class TDLearner:
    """
    TD学習を用いたゲームAIのトレーナー
    """
    
    def __init__(self, 
                 board_size: int = 5,
                 learning_rate: float = 0.0005,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.15,
                 lambda_td: float = 0.7,
                 use_cuda: bool = False):
        """
        Args:
            board_size: ボードのサイズ
            learning_rate: 学習率(より小さく設定)
            discount_factor: 割引率(γ)
            epsilon: ε-greedy方策のε(やや小さく)
            lambda_td: TD(λ)のλ（eligibility trace用）
            use_cuda: CUDAを使用するか
        """
        self.board_size = board_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.lambda_td = lambda_td
        
        # 価値ネットワーク
        self.value_network = ValueNetwork(board_size, learning_rate, use_cuda)
        
        # 学習統計
        self.game_count = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.episode_rewards = []
    
    def select_action(self, game: ContrastGame, use_epsilon: bool = True) -> Optional[Tuple]:
        """
        行動選択（ε-greedy）
        
        Args:
            game: 現在のゲーム状態
            use_epsilon: ε-greedy方策を使用するか
        
        Returns:
            選択された行動 (from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color)
            または None（合法手がない場合）
        """
        legal_moves = self.get_legal_moves(game)
        
        if not legal_moves:
            return None
        
        # ε-greedy方策
        if use_epsilon and random.random() < self.epsilon:
            # ランダムに行動選択
            return random.choice(legal_moves)
        else:
            # 最良の行動を選択
            return self.select_best_move(game, legal_moves)
    
    def get_legal_moves(self, game: ContrastGame) -> List[Tuple]:
        """
        現在の局面での合法手をすべて取得
        
        Returns:
            [(from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color), ...]
        """
        legal_moves = []
        
        # タイル配置のオプション
        tile_options = [(False, None, None, None)]  # タイル配置なし
        
        tiles_remaining = game.tiles_remaining[game.current_player]
        
        # 黒タイルの配置オプション
        if tiles_remaining['black'] > 0:
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if game.board.get_tile_color(x, y) == TileColor.WHITE:
                        tile_options.append((True, x, y, TileColor.BLACK))
        
        # グレータイルの配置オプション
        if tiles_remaining['gray'] > 0:
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if game.board.get_tile_color(x, y) == TileColor.WHITE:
                        tile_options.append((True, x, y, TileColor.GRAY))
        
        # 各コマの移動オプション
        for from_y in range(self.board_size):
            for from_x in range(self.board_size):
                piece = game.board.get_piece(from_x, from_y)
                if piece and piece.owner == game.current_player:
                    # タイルオプションごとに有効な移動を確認
                    for place_tile, tile_x, tile_y, tile_color in tile_options:
                        # 一時的なゲーム状態をコピー
                        temp_game = copy.deepcopy(game)
                        
                        # タイルを配置（該当する場合）
                        if place_tile:
                            temp_game.board.set_tile_color(tile_x, tile_y, tile_color)
                        
                        # 有効な移動先を取得
                        valid_moves = temp_game.get_valid_moves(from_x, from_y)
                        
                        for to_x, to_y in valid_moves:
                            legal_moves.append((
                                from_x, from_y, to_x, to_y,
                                place_tile, tile_x, tile_y, tile_color
                            ))
        
        return legal_moves
    
    def select_best_move(self, game: ContrastGame, legal_moves: List[Tuple]) -> Tuple:
        """
        価値ネットワークを使って最良の手を選択
        
        Args:
            game: 現在のゲーム状態
            legal_moves: 合法手のリスト
        
        Returns:
            最良の手
        """
        best_move = None
        best_value = -float('inf')
        
        current_player = game.current_player
        
        # 各手の評価をバッチ処理で高速化
        move_values = []
        
        for move in legal_moves:
            # 一時的にゲームをコピーして手を実行
            temp_game = copy.deepcopy(game)
            from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color = move
            
            # 手を実行
            temp_game.make_move(from_x, from_y, to_x, to_y, 
                              place_tile, tile_x, tile_y, tile_color)
            
            # 盤面を評価（相手の視点になっているので反転が必要）
            state = board_to_tensor(temp_game)
            value = self.value_network.evaluate(state)
            
            # 手を打った後は相手のターンになっているので、評価値を反転
            value = -value
            
            # 勝利状態にボーナス
            if temp_game.game_over and temp_game.winner == current_player:
                value = 1.0
            
            move_values.append((move, value))
        
        # 最良の手を選択
        best_move, best_value = max(move_values, key=lambda x: x[1])
        
        return best_move
    
    def play_game(self, train: bool = True) -> Tuple[Player, List[Tuple[np.ndarray, float]]]:
        """
        1ゲームをプレイ
        
        Args:
            train: 学習モードかどうか
        
        Returns:
            (勝者, [(状態, 報酬)のリスト])
        """
        game = ContrastGame(self.board_size)
        game.setup_initial_position()
        
        # エピソードの記録
        episode_data = []
        
        move_count = 0
        max_moves = 200  # 無限ループ防止
        
        while not game.game_over and move_count < max_moves:
            # 現在の状態を記録
            current_state = board_to_tensor(game)
            current_player = game.current_player
            
            # 行動選択
            action = self.select_action(game, use_epsilon=train)
            
            if action is None:
                # 合法手がない場合は敗北
                game.game_over = True
                game.winner = Player.PLAYER2 if current_player == Player.PLAYER1 else Player.PLAYER1
                break
            
            # 行動実行
            from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color = action
            game.make_move(from_x, from_y, to_x, to_y, 
                         place_tile, tile_x, tile_y, tile_color)
            
            # 状態と報酬を記録
            episode_data.append((current_state, current_player))
            
            move_count += 1
        
        # ゲーム終了時の報酬を計算
        winner = game.winner
        
        # 最終報酬を各状態に割り当て
        episode_with_rewards = []
        for state, player in episode_data:
            if winner == player:
                reward = 1.0  # 勝利
            elif winner is None:
                reward = 0.0  # 引き分け
            else:
                reward = -1.0  # 敗北
            
            episode_with_rewards.append((state, reward))
        
        return winner, episode_with_rewards
    
    def train_episode(self) -> float:
        """
        1エピソードの学習
        
        Returns:
            平均損失
        """
        # ゲームをプレイ
        winner, episode_data = self.play_game(train=True)
        
        # 統計を更新
        self.game_count += 1
        if winner == Player.PLAYER1:
            self.player1_wins += 1
        elif winner == Player.PLAYER2:
            self.player2_wins += 1
        
        # TD学習を実行
        states = []
        targets = []
        
        # 終端報酬
        final_reward = 1.0 if winner else 0.0
        
        # 逆順に処理してTD誤差を計算
        for i in range(len(episode_data) - 1, -1, -1):
            state, player = episode_data[i]
            
            # 最終ステップの場合
            if i == len(episode_data) - 1:
                if winner == player:
                    target = 1.0
                elif winner is None:
                    target = 0.0
                else:
                    target = -1.0
            else:
                # 次の状態の価値を使用
                next_state, _ = episode_data[i + 1]
                next_value = self.value_network.evaluate(next_state)
                
                # 即時報酬(勝利に近づくと+、遠ざかると-)
                immediate_reward = 0.0
                if winner == player:
                    immediate_reward = 0.01  # 勝利への小さなステップ報酬
                elif winner and winner != player:
                    immediate_reward = -0.01
                
                # TD目標
                target = immediate_reward + self.discount_factor * next_value
            
            states.append(state)
            targets.append(target)
        
        # バッチ学習
        if len(states) > 0:
            loss = self.value_network.train_step(states, targets)
            return loss
        
        return 0.0
    
    def train(self, num_episodes: int, save_interval: int = 100, model_path: str = "contrast_ai.pth"):
        """
        複数エピソードの学習
        
        Args:
            num_episodes: エピソード数
            save_interval: モデル保存間隔
            model_path: モデルの保存パス
        """
        print(f"TD学習を開始します（エピソード数: {num_episodes}）")
        print("="*60)
        
        for episode in range(num_episodes):
            loss = self.train_episode()
            
            # 定期的に統計を表示
            if (episode + 1) % 10 == 0:
                win_rate_p1 = self.player1_wins / self.game_count if self.game_count > 0 else 0
                win_rate_p2 = self.player2_wins / self.game_count if self.game_count > 0 else 0
                
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Loss: {loss:.6f} | "
                      f"P1 Win: {win_rate_p1:.2%} | "
                      f"P2 Win: {win_rate_p2:.2%}")
            
            # 定期的にモデルを保存
            if (episode + 1) % save_interval == 0:
                self.value_network.save(f"{model_path}.ep{episode + 1}")
                print(f"  -> モデルを保存しました")
        
        # 最終モデルを保存
        self.value_network.save(model_path)
        print("\n学習完了！")
        print(f"総ゲーム数: {self.game_count}")
        print(f"Player 1 勝率: {self.player1_wins / self.game_count:.2%}")
        print(f"Player 2 勝率: {self.player2_wins / self.game_count:.2%}")


if __name__ == "__main__":
    # TD学習のデモ
    print("TD学習のデモンストレーション")
    print("="*60)
    
    # 学習器を初期化
    learner = TDLearner(
        board_size=5,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=0.2,
        lambda_td=0.7
    )
    
    # 少数のエピソードで学習テスト
    learner.train(num_episodes=1000, save_interval=100, model_path="contrast_ai_demo.pth")
