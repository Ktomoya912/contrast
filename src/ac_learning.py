"""
Actor-Critic学習アルゴリズム
PolicyとValueを同時に学習してコントラストゲームのAIを訓練
"""

import copy
import logging
from typing import List, Optional, Tuple

import numpy as np

from ai_model import ActorCriticNetwork, board_to_tensor
from contrast_game import ContrastGame, Player, TileColor

# ロギング基本設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ロガー設定
logger = logging.getLogger(__name__)


class ActorCriticLearner:
    """
    Actor-Criticを用いたゲームAIのトレーナー
    """

    def __init__(
        self,
        board_size: int = 5,
        max_actions: int = 500,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        use_cuda: bool = False,
    ):
        """
        Args:
            board_size: ボードのサイズ
            max_actions: 最大行動数
            learning_rate: 学習率
            discount_factor: 割引率（γ）
            use_cuda: CUDAを使用するか
        """
        self.board_size = board_size
        self.max_actions = max_actions
        self.discount_factor = discount_factor

        # Actor-Criticネットワーク
        self.network = ActorCriticNetwork(
            board_size, max_actions, learning_rate, use_cuda
        )

        # 学習統計
        self.game_count = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.episode_rewards = []

    def action_to_tuple(self, action_index: int) -> Optional[Tuple]:
        """
        行動インデックスをゲームの行動タプルに変換

        行動エンコーディング:
        - from_pos (25通り) × to_pos (25通り) × tile_option (5通り) = 3125通り
        - 実際には500通りに制限
        """
        if action_index >= self.max_actions:
            return None

        # 簡易エンコーディング: 最初の500通りを使用
        # from_pos: action // 20
        # to_pos: (action % 20) // 4  (相対位置)
        # tile_option: action % 4

        # より詳細な実装が必要ですが、とりあえず None を返して
        # get_legal_moves から選択する方式にします
        return None

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
        if tiles_remaining["black"] > 0:
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if game.board.get_tile_color(x, y) == TileColor.WHITE:
                        tile_options.append((True, x, y, TileColor.BLACK))

        # グレータイルの配置オプション
        if tiles_remaining["gray"] > 0:
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
                            legal_moves.append(
                                (
                                    from_x,
                                    from_y,
                                    to_x,
                                    to_y,
                                    place_tile,
                                    tile_x,
                                    tile_y,
                                    tile_color,
                                )
                            )

        return legal_moves

    def create_action_mask(self, legal_moves: List[Tuple]) -> np.ndarray:
        """
        合法手マスクを作成

        Args:
            legal_moves: 合法手のリスト

        Returns:
            (max_actions,) のブールマスク
        """
        mask = np.zeros(self.max_actions, dtype=bool)
        # 簡易実装: 最初のN個の行動を合法とする
        num_legal = min(len(legal_moves), self.max_actions)
        mask[:num_legal] = True
        return mask

    def select_action(self, game: ContrastGame) -> Optional[Tuple[int, Tuple]]:
        """
        Policyネットワークを使って行動選択

        Args:
            game: 現在のゲーム状態

        Returns:
            (action_index, action_tuple) または None
        """
        legal_moves = self.get_legal_moves(game)

        if not legal_moves:
            return None

        # 盤面の状態を取得
        state = board_to_tensor(game)
        action_mask = self.create_action_mask(legal_moves)

        # Policyから行動確率を取得
        policy_probs, value = self.network.evaluate(state, action_mask)

        # 合法手の範囲で確率分布を正規化
        num_legal = min(len(legal_moves), self.max_actions)
        legal_probs = policy_probs[:num_legal]
        legal_probs = legal_probs / legal_probs.sum()

        # 確率分布に従ってサンプリング
        action_index = np.random.choice(num_legal, p=legal_probs)

        return action_index, legal_moves[action_index]

    def play_game(self, train: bool = True) -> Tuple[Player, List]:
        """
        1ゲームをプレイ

        Args:
            train: 学習モードかどうか

        Returns:
            (勝者, エピソードデータ)
        """
        game = ContrastGame(self.board_size)
        game.setup_initial_position()

        # エピソードの記録
        episode_data = []  # (state, action_index, action_mask, value, player)

        move_count = 0
        max_moves = 200  # 無限ループ防止

        while not game.game_over and move_count < max_moves:
            # 現在の状態を記録
            current_state = board_to_tensor(game)
            current_player = game.current_player

            # 行動選択
            action_result = self.select_action(game)

            if action_result is None:
                # 合法手がない場合は敗北
                game.game_over = True
                game.winner = (
                    Player.PLAYER2
                    if current_player == Player.PLAYER1
                    else Player.PLAYER1
                )
                break

            action_index, action_tuple = action_result
            from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color = (
                action_tuple
            )

            # 合法手マスクを取得
            legal_moves = self.get_legal_moves(game)
            action_mask = self.create_action_mask(legal_moves)

            # 現在の価値を取得
            _, value = self.network.evaluate(current_state, action_mask)

            # 行動実行
            game.make_move(
                from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
            )

            # 状態、行動、マスク、価値を記録
            episode_data.append(
                (current_state, action_index, action_mask, value, current_player)
            )

            move_count += 1

        winner = game.winner

        return winner, episode_data

    def train_episode(self) -> Tuple[float, float, float]:
        """
        1エピソードの学習

        Returns:
            (policy_loss, value_loss, total_loss)
        """
        # ゲームをプレイ
        winner, episode_data = self.play_game(train=True)

        # 統計を更新
        self.game_count += 1
        if winner == Player.PLAYER1:
            self.player1_wins += 1
        elif winner == Player.PLAYER2:
            self.player2_wins += 1

        if len(episode_data) == 0:
            return 0.0, 0.0, 0.0

        # Advantageと目標値を計算
        states = []
        action_indices = []
        action_masks = []
        advantages = []
        value_targets = []

        # 終端報酬
        final_reward = 0.0

        # 逆順に処理してリターンを計算
        G = 0.0
        for i in range(len(episode_data) - 1, -1, -1):
            state, action_idx, action_mask, value, player = episode_data[i]

            # 報酬の計算
            if i == len(episode_data) - 1:
                # 最終ステップ
                if winner == player:
                    reward = 1.0
                elif winner is None:
                    reward = 0.0
                else:
                    reward = -1.0
            else:
                # 中間ステップ: 小さなステップ報酬
                reward = 0.0

            # リターン（累積報酬）
            G = reward + self.discount_factor * G

            # Advantage = G - V(s)
            advantage = G - value

            states.append(state)
            action_indices.append(action_idx)
            action_masks.append(action_mask)
            advantages.append(advantage)
            value_targets.append(G)

        # 順序を戻す
        states.reverse()
        action_indices.reverse()
        action_masks.reverse()
        advantages.reverse()
        value_targets.reverse()

        # 学習
        if len(states) > 0:
            p_loss, v_loss, t_loss = self.network.train_step(
                states, action_indices, action_masks, advantages, value_targets
            )
            return p_loss, v_loss, t_loss

        return 0.0, 0.0, 0.0

    def train(
        self,
        num_episodes: int,
        save_interval: int = 100,
        model_path: str = "contrast_ac.pth",
    ):
        """
        複数エピソードの学習

        Args:
            num_episodes: エピソード数
            save_interval: モデル保存間隔
            model_path: モデルの保存パス
        """
        logger.info(f"Actor-Critic学習を開始します（エピソード数: {num_episodes}）")
        logger.info("=" * 60)

        for episode in range(num_episodes):
            p_loss, v_loss, t_loss = self.train_episode()

            # 定期的に統計を表示
            if (episode + 1) % 10 == 0:
                win_rate_p1 = (
                    self.player1_wins / self.game_count if self.game_count > 0 else 0
                )
                win_rate_p2 = (
                    self.player2_wins / self.game_count if self.game_count > 0 else 0
                )

                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"P_Loss: {p_loss:.4f} | V_Loss: {v_loss:.4f} | T_Loss: {t_loss:.4f} | "
                    f"P1: {win_rate_p1:.2%} | P2: {win_rate_p2:.2%}"
                )

            # 定期的にモデルを保存
            if (episode + 1) % save_interval == 0:
                self.network.save(f"{model_path}.ep{episode + 1}")
                logger.info("  -> モデルを保存しました")

        # 最終モデルを保存
        self.network.save(model_path)
        logger.info("\n学習完了！")
        logger.info(f"総ゲーム数: {self.game_count}")
        logger.info(f"Player 1 勝率: {self.player1_wins / self.game_count:.2%}")
        logger.info(f"Player 2 勝率: {self.player2_wins / self.game_count:.2%}")


if __name__ == "__main__":
    # Actor-Critic学習のデモンストレーション
    logger.info("Actor-Critic学習のデモンストレーション")
    logger.info("=" * 60)

    # 学習器を初期化
    learner = ActorCriticLearner(
        board_size=5,
        max_actions=500,
        learning_rate=0.0003,
        discount_factor=0.99,
        use_cuda=False,
    )

    # 少数のエピソードで学習テスト
    learner.train(num_episodes=100, save_interval=50, model_path="contrast_ac_demo.pth")
