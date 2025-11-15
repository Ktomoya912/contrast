"""
敵対的学習（ルールベース相手）と自己対戦を交互に行う学習
"""

import argparse
import logging

from ac_learning import ActorCriticLearner
from ai_model import board_to_tensor
from contrast_game import ContrastGame, Player
from rule_based_player import RuleBasedPlayer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MixedLearner(ActorCriticLearner):
    """敵対的学習と自己対戦を混合する学習器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule_player_p1 = RuleBasedPlayer(Player.PLAYER1)
        self.rule_player_p2 = RuleBasedPlayer(Player.PLAYER2)

    def play_game_vs_rule(self, ai_player: Player, train: bool = True):
        """
        AIとルールベースプレイヤーの対戦

        Args:
            ai_player: AIが担当するプレイヤー
            train: 学習モードかどうか

        Returns:
            (勝者, エピソードデータ)
        """
        game = ContrastGame(self.board_size)
        game.setup_initial_position()

        # エピソードの記録（AIの行動のみ）
        episode_data = []

        move_count = 0
        max_moves = 200

        rule_player = (
            self.rule_player_p2 if ai_player == Player.PLAYER1 else self.rule_player_p1
        )

        while not game.game_over and move_count < max_moves:
            current_player = game.current_player

            if current_player == ai_player:
                # AIのターン
                current_state = board_to_tensor(game)

                action_result = self.select_action(game)

                if action_result is None:
                    game.game_over = True
                    if current_player == Player.PLAYER1:
                        game.winner = Player.PLAYER2
                    else:
                        game.winner = Player.PLAYER1
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
                success = game.make_move(
                    from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
                )

                if not success:
                    game.game_over = True
                    game.winner = (
                        Player.PLAYER2
                        if current_player == Player.PLAYER1
                        else Player.PLAYER1
                    )
                    break

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

            else:
                # ルールベースのターン
                action = rule_player.select_action(game)

                if action is None:
                    game.game_over = True
                    game.winner = ai_player
                    break

                move_tuple, tile_tuple = action
                from_x, from_y, to_x, to_y = move_tuple

                if tile_tuple:
                    place_tile = True
                    tile_color, tile_x, tile_y = tile_tuple
                else:
                    place_tile = False
                    tile_x, tile_y, tile_color = None, None, None

                success = game.make_move(
                    from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
                )

                if not success:
                    game.game_over = True
                    game.winner = ai_player
                    break

            move_count += 1

        winner = game.winner

        return winner, episode_data

    def train_episode_vs_rule(self, ai_player: Player):
        """
        ルールベース相手に1エピソードの学習

        Args:
            ai_player: AIが担当するプレイヤー

        Returns:
            (move_loss, tile_loss, value_loss, total_loss, winner)
        """
        # ゲームをプレイ
        winner, episode_data = self.play_game_vs_rule(ai_player, train=True)

        # 統計を更新
        self.game_count += 1
        if winner == Player.PLAYER1:
            self.player1_wins += 1
        elif winner == Player.PLAYER2:
            self.player2_wins += 1

        if len(episode_data) == 0:
            return 0.0, 0.0, 0.0, 0.0, winner

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
            else:
                # 中間ステップ：前進報酬
                reward = 0.0

                # 前進したかチェック
                if i + 1 < len(episode_data):
                    next_move_idx = episode_data[i][1]
                    from_x, from_y, direction = self.index_to_move(next_move_idx)

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
                    if player == Player.PLAYER1:
                        if dy < 0:
                            reward = 0.05
                    else:
                        if dy > 0:
                            reward = 0.05

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
        move_loss, tile_loss, value_loss, total_loss = self.network.train_step(
            states,
            move_indices,
            tile_indices,
            move_masks,
            tile_masks,
            advantages,
            value_targets,
        )

        return move_loss, tile_loss, value_loss, total_loss, winner

    def train_mixed(
        self,
        num_cycles: int,
        episodes_per_cycle: int,
        save_interval: int,
        model_path: str,
    ):
        """
        敵対的学習と自己対戦を交互に実行

        Args:
            num_cycles: サイクル数
            episodes_per_cycle: 1サイクルあたりのエピソード数
            save_interval: モデル保存間隔（エピソード単位）
            model_path: モデル保存パス
        """
        logger.info(
            f"混合学習を開始します（サイクル数: {num_cycles}, サイクルあたり: {episodes_per_cycle}エピソード）"
        )
        logger.info("=" * 60)

        total_episodes = 0

        for cycle in range(num_cycles):
            # 交互に学習モードを切り替え
            if cycle % 2 == 0:
                # 敵対的学習（ルールベース相手）
                mode = "敵対的学習"
                logger.info(f"\n【サイクル {cycle + 1}/{num_cycles}】{mode}")
                logger.info("-" * 60)

                for ep in range(episodes_per_cycle):
                    total_episodes += 1

                    # AIの先後をランダムに決定
                    ai_player = Player.PLAYER1 if ep % 2 == 0 else Player.PLAYER2

                    move_loss, tile_loss, value_loss, total_loss, winner = (
                        self.train_episode_vs_rule(ai_player)
                    )

                    # 10エピソードごとに進捗表示
                    if (ep + 1) % 10 == 0 or ep == 0:
                        p1_rate = (
                            self.player1_wins / self.game_count * 100
                            if self.game_count > 0
                            else 0
                        )
                        p2_rate = (
                            self.player2_wins / self.game_count * 100
                            if self.game_count > 0
                            else 0
                        )
                        ai_win = (winner == ai_player) if winner else False
                        logger.info(
                            f"Ep {total_episodes} ({mode}) | M_Loss: {move_loss:.4f} | "
                            f"T_Loss: {tile_loss:.4f} | V_Loss: {value_loss:.4f} | "
                            f"Total: {total_loss:.4f} | AI Win: {ai_win} | "
                            f"P1: {p1_rate:.1f}% | P2: {p2_rate:.1f}%"
                        )

                    # モデル保存
                    if total_episodes % save_interval == 0:
                        self.network.save(f"{model_path}.ep{total_episodes}")
                        logger.info(f"  -> モデルを保存しました: ep{total_episodes}")

            else:
                # 自己対戦学習
                mode = "自己対戦学習"
                logger.info(f"\n【サイクル {cycle + 1}/{num_cycles}】{mode}")
                logger.info("-" * 60)

                for ep in range(episodes_per_cycle):
                    total_episodes += 1

                    move_loss, tile_loss, value_loss, total_loss = self.train_episode()

                    # 10エピソードごとに進捗表示
                    if (ep + 1) % 10 == 0 or ep == 0:
                        p1_rate = (
                            self.player1_wins / self.game_count * 100
                            if self.game_count > 0
                            else 0
                        )
                        p2_rate = (
                            self.player2_wins / self.game_count * 100
                            if self.game_count > 0
                            else 0
                        )
                        logger.info(
                            f"Ep {total_episodes} ({mode}) | M_Loss: {move_loss:.4f} | "
                            f"T_Loss: {tile_loss:.4f} | V_Loss: {value_loss:.4f} | "
                            f"Total: {total_loss:.4f} | "
                            f"P1: {p1_rate:.1f}% | P2: {p2_rate:.1f}%"
                        )

                    # モデル保存
                    if total_episodes % save_interval == 0:
                        self.network.save(f"{model_path}.ep{total_episodes}")
                        logger.info(f"  -> モデルを保存しました: ep{total_episodes}")

        # 最終モデルを保存
        self.network.save(model_path)
        logger.info("\n学習完了！")
        logger.info(f"総エピソード数: {total_episodes}")
        logger.info(f"総ゲーム数: {self.game_count}")
        logger.info(f"Player 1 勝率: {self.player1_wins / self.game_count * 100:.2f}%")
        logger.info(f"Player 2 勝率: {self.player2_wins / self.game_count * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="混合学習（敵対的 + 自己対戦）")
    parser.add_argument(
        "--cycles", type=int, default=5, help="サイクル数 (デフォルト: 5)"
    )
    parser.add_argument(
        "--episodes-per-cycle",
        type=int,
        default=200,
        help="1サイクルあたりのエピソード数 (デフォルト: 200)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="モデル保存間隔 (デフォルト: 100)",
    )
    parser.add_argument(
        "--model", type=str, default="contrast_ac_mixed.pth", help="モデル保存パス"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="読み込むモデルパス（継続学習）"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="学習率 (デフォルト: 0.0003)"
    )
    parser.add_argument(
        "--discount", type=float, default=0.99, help="割引率 (デフォルト: 0.99)"
    )
    parser.add_argument("--cuda", action="store_true", help="CUDAを使用")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Contrast AI - 混合学習（敵対的 + 自己対戦）")
    logger.info("=" * 60)
    logger.info(f"サイクル数: {args.cycles}")
    logger.info(f"サイクルあたりエピソード数: {args.episodes_per_cycle}")
    logger.info(f"総エピソード数: {args.cycles * args.episodes_per_cycle}")
    logger.info(f"学習率: {args.lr}")
    logger.info(f"割引率: {args.discount}")
    logger.info(f"保存間隔: {args.save_interval}")
    logger.info(f"モデルパス: {args.model}")
    logger.info(f"CUDA: {args.cuda}")
    logger.info("=" * 60)
    logger.info("")

    # 学習器を初期化
    learner = MixedLearner(
        board_size=5,
        max_actions=500,
        learning_rate=args.lr,
        discount_factor=args.discount,
        use_cuda=args.cuda,
    )

    # 既存モデルを読み込み（継続学習）
    if args.load:
        try:
            learner.network.load(args.load)
            logger.info(f"モデルを読み込みました: {args.load}\n")
        except Exception as e:
            logger.warning(f"モデル読み込みエラー: {e}. 新規学習を開始します\n")

    # 混合学習実行
    learner.train_mixed(
        num_cycles=args.cycles,
        episodes_per_cycle=args.episodes_per_cycle,
        save_interval=args.save_interval,
        model_path=args.model,
    )

    logger.info("=" * 60)
    logger.info(f"モデルは {args.model} に保存されました")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
