"""
AIとルールベースプレイヤーの対戦
"""

import argparse
import logging

from ac_learning import ActorCriticLearner
from contrast_game import ContrastGame, Player
from rule_based_player import RuleBasedPlayer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def play_game(
    ai_player: Player,
    ai_learner: ActorCriticLearner,
    rule_player: RuleBasedPlayer,
    verbose: bool = True,
) -> Player:
    """
    1ゲームを実行

    Args:
        ai_player: AIが担当するプレイヤー
        ai_learner: AI学習器
        rule_player: ルールベースプレイヤー
        verbose: 詳細表示

    Returns:
        勝者
    """
    game = ContrastGame(board_size=5)
    game.setup_initial_position()

    move_count = 0
    max_moves = 200

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"AI: Player {ai_player.value}, Rule-based: Player {rule_player.player.value}"
        )
        print(f"{'=' * 60}")

    while not game.game_over and move_count < max_moves:
        current_player = game.current_player

        if current_player == ai_player:
            # AIのターン
            action = ai_learner.select_action(game)
            if action is None:
                if verbose:
                    print(f"AI (Player {current_player.value}) has no valid moves!")
                break

            move_index, tile_index, move_tuple, tile_tuple = action
            from_x, from_y, to_x, to_y = move_tuple

            if tile_tuple:
                place_tile = True
                tile_color, tile_x, tile_y = tile_tuple
                if verbose:
                    print(
                        f"AI P{current_player.value}: ({from_x},{from_y}) -> ({to_x},{to_y}), "
                        f"Tile: {tile_color.value} at ({tile_x},{tile_y})"
                    )
            else:
                place_tile = False
                tile_x, tile_y, tile_color = None, None, None
                if verbose:
                    print(
                        f"AI P{current_player.value}: ({from_x},{from_y}) -> ({to_x},{to_y})"
                    )

        else:
            # ルールベースのターン
            action = rule_player.select_action(game)
            if action is None:
                if verbose:
                    print(
                        f"Rule-based (Player {current_player.value}) has no valid moves!"
                    )
                break

            move_tuple, tile_tuple = action
            from_x, from_y, to_x, to_y = move_tuple

            if tile_tuple:
                place_tile = True
                tile_color, tile_x, tile_y = tile_tuple
                if verbose:
                    print(
                        f"Rule P{current_player.value}: ({from_x},{from_y}) -> ({to_x},{to_y}), "
                        f"Tile: {tile_color.value} at ({tile_x},{tile_y})"
                    )
            else:
                place_tile = False
                tile_x, tile_y, tile_color = None, None, None
                if verbose:
                    print(
                        f"Rule P{current_player.value}: ({from_x},{from_y}) -> ({to_x},{to_y})"
                    )

        # 手を実行
        success = game.make_move(
            from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
        )

        if not success:
            if verbose:
                print(f"Invalid move by Player {current_player.value}!")
            break

        move_count += 1

    if verbose:
        print(f"{'=' * 60}")
        if game.game_over:
            winner_type = "AI" if game.winner == ai_player else "Rule-based"
            print(f"Game Over! Winner: Player {game.winner.value} ({winner_type})")
        else:
            print(f"Max moves ({max_moves}) reached - Draw")
        print(f"Total moves: {move_count}")
        print(f"{'=' * 60}\n")

    return game.winner


def battle(
    num_games: int,
    model_path: str = "contrast_ac.pth",
    ai_side: str = "both",
    verbose: bool = False,
):
    """
    複数回の対戦を実行

    Args:
        num_games: 対戦回数
        model_path: AIモデルのパス
        ai_side: AIの担当 ('1', '2', 'both')
        verbose: 各ゲームの詳細を表示
    """
    # AI学習器を初期化
    ai_learner = ActorCriticLearner(
        board_size=5,
        max_actions=500,
        learning_rate=0.0003,
        discount_factor=0.99,
        use_cuda=False,
    )

    # モデルを読み込み
    try:
        ai_learner.network.load(model_path)
        logger.info(f"AIモデルを読み込みました: {model_path}")
    except Exception as e:
        logger.warning(f"モデル読み込みエラー: {e}. 未学習AIを使用します")

    # 統計
    results = {
        "ai_wins": 0,
        "rule_wins": 0,
        "draws": 0,
        "ai_as_p1_wins": 0,
        "ai_as_p2_wins": 0,
        "rule_as_p1_wins": 0,
        "rule_as_p2_wins": 0,
    }

    print("\n" + "=" * 60)
    print(f"対戦開始: AI vs ルールベース ({num_games}ゲーム)")
    print("=" * 60)

    for game_num in range(num_games):
        # AIとルールベースの先後を決定
        if ai_side == "1":
            ai_player = Player.PLAYER1
            rule_player_obj = RuleBasedPlayer(Player.PLAYER2)
        elif ai_side == "2":
            ai_player = Player.PLAYER2
            rule_player_obj = RuleBasedPlayer(Player.PLAYER1)
        else:  # both
            if game_num % 2 == 0:
                ai_player = Player.PLAYER1
                rule_player_obj = RuleBasedPlayer(Player.PLAYER2)
            else:
                ai_player = Player.PLAYER2
                rule_player_obj = RuleBasedPlayer(Player.PLAYER1)

        if not verbose:
            if (game_num + 1) % 10 == 0 or game_num == 0:
                print(f"Game {game_num + 1}/{num_games}...", end=" ", flush=True)

        winner = play_game(ai_player, ai_learner, rule_player_obj, verbose)

        # 結果を記録
        if winner is None:
            results["draws"] += 1
            if not verbose and ((game_num + 1) % 10 == 0 or game_num == 0):
                print("Draw")
        elif winner == ai_player:
            results["ai_wins"] += 1
            if winner == Player.PLAYER1:
                results["ai_as_p1_wins"] += 1
            else:
                results["ai_as_p2_wins"] += 1
            if not verbose and ((game_num + 1) % 10 == 0 or game_num == 0):
                print(f"AI Win (as P{winner.value})")
        else:
            results["rule_wins"] += 1
            if winner == Player.PLAYER1:
                results["rule_as_p1_wins"] += 1
            else:
                results["rule_as_p2_wins"] += 1
            if not verbose and ((game_num + 1) % 10 == 0 or game_num == 0):
                print(f"Rule Win (as P{winner.value})")

    # 結果表示
    print("\n" + "=" * 60)
    print("対戦結果")
    print("=" * 60)
    print(f"総ゲーム数: {num_games}")
    print(f"AI勝利: {results['ai_wins']} ({results['ai_wins'] / num_games * 100:.1f}%)")
    print(f"  - Player 1として: {results['ai_as_p1_wins']} 勝")
    print(f"  - Player 2として: {results['ai_as_p2_wins']} 勝")
    print(
        f"ルールベース勝利: {results['rule_wins']} ({results['rule_wins'] / num_games * 100:.1f}%)"
    )
    print(f"  - Player 1として: {results['rule_as_p1_wins']} 勝")
    print(f"  - Player 2として: {results['rule_as_p2_wins']} 勝")
    print(f"引き分け: {results['draws']} ({results['draws'] / num_games * 100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI vs ルールベースプレイヤーの対戦")
    parser.add_argument(
        "--games", type=int, default=50, help="対戦回数 (デフォルト: 50)"
    )
    parser.add_argument(
        "--model", type=str, default="contrast_ac.pth", help="AIモデルのパス"
    )
    parser.add_argument(
        "--ai-side",
        type=str,
        choices=["1", "2", "both"],
        default="both",
        help="AIの担当プレイヤー (1, 2, both)",
    )
    parser.add_argument("--verbose", action="store_true", help="各ゲームの詳細を表示")

    args = parser.parse_args()

    battle(args.games, args.model, args.ai_side, args.verbose)
