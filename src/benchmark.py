"""
NumPy最適化のパフォーマンステスト
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.contrast_game import ContrastGame, Player
from src.rule_based_player import RuleBasedPlayer


def benchmark_games(num_games=100):
    """指定回数のゲームを実行してパフォーマンスを測定"""
    print(f"ベンチマーク: {num_games}ゲームのルールベースAI対戦")
    print("=" * 50)

    rule_player1 = RuleBasedPlayer(Player.PLAYER1)
    rule_player2 = RuleBasedPlayer(Player.PLAYER2)

    start_time = time.time()

    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_moves = 0

    for game_num in range(num_games):
        game = ContrastGame(board_size=5)
        game.setup_initial_position()

        move_count = 0
        max_moves = 100

        while not game.game_over and move_count < max_moves:
            current_player = game.current_player

            if current_player == Player.PLAYER1:
                action = rule_player1.select_action(game)
            else:
                action = rule_player2.select_action(game)

            if not action:
                break

            move_tuple, tile_tuple = action
            from_x, from_y, to_x, to_y = move_tuple

            if tile_tuple is None:
                place_tile = False
                tile_x, tile_y, tile_color = None, None, None
            else:
                place_tile = True
                tile_color, tile_x, tile_y = tile_tuple

            success = game.make_move(
                from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
            )

            if not success:
                break

            move_count += 1

        total_moves += move_count

        if game.winner == Player.PLAYER1:
            p1_wins += 1
        elif game.winner == Player.PLAYER2:
            p2_wins += 1
        else:
            draws += 1

        if (game_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (game_num + 1) / elapsed
            print(f"  {game_num + 1}ゲーム完了 ({rate:.2f} games/sec)")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print("=" * 50)
    print("ベンチマーク結果:")
    print(f"  総ゲーム数: {num_games}")
    print(f"  実行時間: {elapsed_time:.2f}秒")
    print(f"  平均速度: {num_games / elapsed_time:.2f} games/sec")
    print(f"  平均手数: {total_moves / num_games:.1f} moves/game")
    print()
    print("勝敗:")
    print(f"  Player 1: {p1_wins} ({p1_wins / num_games * 100:.1f}%)")
    print(f"  Player 2: {p2_wins} ({p2_wins / num_games * 100:.1f}%)")
    print(f"  引き分け: {draws} ({draws / num_games * 100:.1f}%)")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NumPy最適化のベンチマーク")
    parser.add_argument(
        "--games", type=int, default=100, help="実行するゲーム数（デフォルト: 100）"
    )

    args = parser.parse_args()

    benchmark_games(args.games)
