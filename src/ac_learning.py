"""
Actor-Critic学習アルゴリズム
PolicyとValueを同時に学習してコントラストゲームのAIを訓練
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ai_model import ActorCriticNetwork, board_to_tensor
from contrast_game import ContrastGame, Player, TileColor

# ロガー設定
logger = logging.getLogger(__name__)


class ActorCriticLearner:
    """
    Actor-Criticを用いたゲームAIのトレーナー
    """

    def __init__(
        self,
        board_size: int = 5,
        max_actions: int = 500,  # 互換性のため残すが使用しない
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        use_cuda: bool = False,
    ):
        """
        Args:
            board_size: ボードのサイズ
            max_actions: 互換性のため残すが使用しない
            learning_rate: 学習率
            discount_factor: 割引率（γ）
            use_cuda: CUDAを使用するか
        """
        self.board_size = board_size
        self.discount_factor = discount_factor

        # 新しい行動空間
        self.max_move_actions = board_size * board_size * 8  # 200
        self.max_tile_actions = 31  # なし1 + 黒15 + グレー15

        # Actor-Criticネットワーク
        self.network = ActorCriticNetwork(
            board_size, max_actions, learning_rate, use_cuda
        )

        # 学習統計
        self.game_count = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.episode_rewards = []

    def move_to_index(self, from_x: int, from_y: int, direction: int) -> int:
        """
        移動をインデックスに変換

        Args:
            from_x, from_y: 移動元の座標
            direction: 移動方向 (0-7: 8方向)

        Returns:
            インデックス (0-199)
        """
        return from_y * self.board_size * 8 + from_x * 8 + direction

    def index_to_move(self, index: int) -> Tuple[int, int, int]:
        """
        インデックスを移動に変換

        Returns:
            (from_x, from_y, direction)
        """
        direction = index % 8
        index //= 8
        from_x = index % self.board_size
        from_y = index // self.board_size
        return from_x, from_y, direction

    def tile_to_index(self, tile_action: Optional[Tuple]) -> int:
        """
        タイル配置をインデックスに変換

        Args:
            tile_action: None または (tile_color, x, y)

        Returns:
            インデックス (0=なし, 1-15=黒, 16-30=グレー)
        """
        if tile_action is None or tile_action[0] is None:
            return 0

        tile_color, x, y = tile_action
        position_index = y * self.board_size + x

        # 15箇所に制限
        if position_index >= 15:
            position_index = position_index % 15

        if tile_color == TileColor.BLACK:
            return 1 + position_index
        elif tile_color == TileColor.GRAY:
            return 16 + position_index
        else:
            return 0

    def index_to_tile(self, index: int) -> Optional[Tuple]:
        """
        インデックスをタイル配置に変換

        Returns:
            None または (tile_color, x, y)
        """
        if index == 0:
            return None

        if 1 <= index <= 15:
            position_index = index - 1
            tile_color = TileColor.BLACK
        elif 16 <= index <= 30:
            position_index = index - 16
            tile_color = TileColor.GRAY
        else:
            return None

        x = position_index % self.board_size
        y = position_index // self.board_size
        return (tile_color, x, y)

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

    def get_legal_moves(self, game: ContrastGame) -> List[Tuple[int, int, int, int]]:
        """
        合法な移動のリストを取得（タイル配置は別途）

        Returns:
            [(from_x, from_y, to_x, to_y), ...]
        """
        legal_moves = []

        for from_y in range(self.board_size):
            for from_x in range(self.board_size):
                piece = game.board.get_piece(from_x, from_y)
                if piece and piece.owner == game.current_player:
                    # 有効な移動先を取得
                    valid_moves = game.get_valid_moves(from_x, from_y)

                    for to_x, to_y in valid_moves:
                        legal_moves.append((from_x, from_y, to_x, to_y))

        return legal_moves

    def get_legal_tiles(self, game: ContrastGame) -> List[int]:
        """
        合法なタイル配置のインデックスリストを取得

        Returns:
            [0(なし), 1-15(黒), 16-30(グレー)]
        """
        legal_tiles = [0]  # タイル配置なしは常に合法

        tiles_remaining = game.tiles_remaining[game.current_player]

        # 盤面上の空きマスを確認
        for y in range(self.board_size):
            for x in range(self.board_size):
                # コマがある場所にはタイルを置けない
                if game.board.get_piece(x, y) is not None:
                    continue

                # 既存のタイル（白以外）がある場所には置けない
                if game.board.get_tile_color(x, y) != TileColor.WHITE:
                    continue

                position_index = y * self.board_size + x
                if position_index >= 15:
                    position_index = position_index % 15

                # 黒タイルが残っていれば追加
                if tiles_remaining["black"] > 0:
                    legal_tiles.append(1 + position_index)

                # グレータイルが残っていれば追加
                if tiles_remaining["gray"] > 0:
                    legal_tiles.append(16 + position_index)

        return list(set(legal_tiles))  # 重複削除

    def create_move_mask(
        self, legal_moves: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        移動の合法手マスクを作成

        Returns:
            (max_move_actions,) のブールマスク
        """
        mask = np.zeros(self.max_move_actions, dtype=bool)

        # 8方向ベクトル
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for from_x, from_y, to_x, to_y in legal_moves:
            # 移動方向を判定
            dx = np.sign(to_x - from_x)
            dy = np.sign(to_y - from_y)

            try:
                direction = directions.index((dx, dy))
                index = self.move_to_index(from_x, from_y, direction)
                if 0 <= index < self.max_move_actions:
                    mask[index] = True
            except ValueError:
                # 方向が見つからない場合はスキップ
                continue

        return mask

    def create_tile_mask(self, legal_tiles: List[int]) -> np.ndarray:
        """
        タイル配置の合法手マスクを作成

        Returns:
            (max_tile_actions,) のブールマスク
        """
        mask = np.zeros(self.max_tile_actions, dtype=bool)
        for index in legal_tiles:
            if 0 <= index < self.max_tile_actions:
                mask[index] = True
        return mask

    def select_action(
        self, game: ContrastGame
    ) -> Optional[Tuple[int, int, Tuple, Tuple]]:
        """
        移動PolicyとタイルPolicyを使って行動選択

        Args:
            game: 現在のゲーム状態

        Returns:
            (move_index, tile_index, move_tuple, tile_tuple) または None
            move_tuple: (from_x, from_y, to_x, to_y)
            tile_tuple: None or (tile_color, x, y)
        """
        legal_moves = self.get_legal_moves(game)
        legal_tiles = self.get_legal_tiles(game)

        if not legal_moves:
            return None

        # 盤面の状態を取得
        state = board_to_tensor(game)
        move_mask = self.create_move_mask(legal_moves)
        tile_mask = self.create_tile_mask(legal_tiles)

        # Policyから行動確率を取得
        move_probs, tile_probs, value = self.network.evaluate(
            state, move_mask, tile_mask
        )

        # 移動を選択
        move_indices = np.where(move_mask)[0]
        if len(move_indices) == 0:
            return None

        move_probs_masked = move_probs[move_mask]
        if move_probs_masked.sum() > 0:
            move_probs_masked = move_probs_masked / move_probs_masked.sum()
        else:
            move_probs_masked = np.ones(len(move_probs_masked)) / len(move_probs_masked)

        move_index = np.random.choice(move_indices, p=move_probs_masked)

        # タイル配置を選択
        tile_probs_masked = tile_probs[tile_mask]
        if tile_probs_masked.sum() > 0:
            tile_probs_masked = tile_probs_masked / tile_probs_masked.sum()
        else:
            tile_probs_masked = np.ones(len(tile_probs_masked)) / len(tile_probs_masked)

        tile_index = np.random.choice(legal_tiles, p=tile_probs_masked)

        # インデックスを行動に変換
        # 移動は8方向なので実際の移動先を計算
        from_x, from_y, direction = self.index_to_move(move_index)
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        dx, dy = directions[direction]

        # legal_movesから対応する移動を見つける
        move_tuple = None
        for lm in legal_moves:
            if lm[0] == from_x and lm[1] == from_y:
                lm_dx = np.sign(lm[2] - lm[0])
                lm_dy = np.sign(lm[3] - lm[1])
                if lm_dx == dx and lm_dy == dy:
                    move_tuple = lm
                    break

        if move_tuple is None:
            # 見つからない場合は最初の合法手を使う
            move_tuple = legal_moves[0]

        tile_tuple = self.index_to_tile(tile_index)

        return move_index, tile_index, move_tuple, tile_tuple

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
        episode_data = []  # (state, move_index, tile_index, move_mask, tile_mask, value, player)

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

            move_index, tile_index, move_tuple, tile_tuple = action_result
            from_x, from_y, to_x, to_y = move_tuple

            # 合法手マスクを取得
            legal_moves = self.get_legal_moves(game)
            legal_tiles = self.get_legal_tiles(game)
            move_mask = self.create_move_mask(legal_moves)
            tile_mask = self.create_tile_mask(legal_tiles)

            # 現在の価値を取得
            _, _, value = self.network.evaluate(current_state, move_mask, tile_mask)

            # タイル配置情報を準備
            if tile_tuple is None:
                place_tile = False
                tile_x, tile_y, tile_color = None, None, None
            else:
                place_tile = True
                tile_color, tile_x, tile_y = tile_tuple

            # 行動実行
            game.make_move(
                from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
            )

            # 状態、行動、マスク、価値を記録
            episode_data.append(
                (
                    current_state,
                    move_index,
                    tile_index,
                    move_mask,
                    tile_mask,
                    value,
                    current_player,
                )
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
        move_indices = []
        tile_indices = []
        move_masks = []
        tile_masks = []
        advantages = []
        value_targets = []

        # 逆順に処理してリターンを計算
        G = 0.0
        for i in range(len(episode_data) - 1, -1, -1):
            state, move_idx, tile_idx, move_mask, tile_mask, value, player = (
                episode_data[i]
            )

            # 報酬の計算
            if i == len(episode_data) - 1:
                # 最終ステップ
                if winner == player:
                    reward = 1.0
                elif winner is None:
                    reward = 0.0
                else:
                    reward = -1.0

            # リターン（累積報酬）
            G = reward + self.discount_factor * G

            # Advantage = G - V(s)
            advantage = G - value

            states.append(state)
            move_indices.append(move_idx)
            tile_indices.append(tile_idx)
            move_masks.append(move_mask)
            tile_masks.append(tile_mask)
            advantages.append(advantage)
            value_targets.append(G)

        # 順序を戻す
        states.reverse()
        move_indices.reverse()
        tile_indices.reverse()
        move_masks.reverse()
        tile_masks.reverse()
        advantages.reverse()
        value_targets.reverse()

        # 学習
        if len(states) > 0:
            m_loss, t_loss, v_loss, total_loss = self.network.train_step(
                states,
                move_indices,
                tile_indices,
                move_masks,
                tile_masks,
                advantages,
                value_targets,
            )
            return m_loss, t_loss, v_loss, total_loss

        return 0.0, 0.0, 0.0, 0.0

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
            m_loss, t_loss, v_loss, total_loss = self.train_episode()

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
                    f"M_Loss: {m_loss:.4f} | T_Loss: {t_loss:.4f} | V_Loss: {v_loss:.4f} | Total: {total_loss:.4f} | "
                    f"P1: {win_rate_p1:.2%} | P2: {win_rate_p2:.2%}"
                )

            # 定期的にモデルを保存
            if (episode + 1) % save_interval == 0:
                self.network.save(f"{model_path}.ep{episode + 1}")
                logger.info("  -> モデルを保存しました")

        # 最終モデルを保存
        self.network.save(model_path)
        logger.info("学習完了！")
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
