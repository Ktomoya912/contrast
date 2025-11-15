"""
コントラストゲームのルールベースプレイヤー
ヒューリスティックに基づいた戦略的なプレイを行う
"""

import random
from typing import List, Optional, Tuple

from contrast_game import ContrastGame, Player, TileColor


class RuleBasedPlayer:
    """
    ルールベースの戦略的プレイヤー

    戦略の優先順位:
    1. 勝利: ゴールに到達できるなら即座に実行
    2. 防御: 相手のゴール到達を妨害
    3. 前進: ゴールに向かって最も進める手を選択
    4. タイル配置: 自分に有利、相手に不利なタイル配置
    """

    def __init__(self, player: Player):
        self.player = player

    def select_action(
        self, game: ContrastGame
    ) -> Optional[Tuple[Tuple[int, int, int, int], Optional[Tuple]]]:
        """
        最適な行動を選択

        Returns:
            move_tuple: (from_x, from_y, to_x, to_y)
            tile_tuple: None or (tile_color, x, y)
        """
        if game.current_player != self.player:
            return None

        # すべての合法な移動を取得
        legal_moves = self._get_all_legal_moves(game)

        if not legal_moves:
            return None

        # 1. 勝利できる手があるか確認
        winning_move = self._find_winning_move(game, legal_moves)
        if winning_move:
            tile_action = self._select_tile_placement(game, winning_move)
            return (winning_move, tile_action)

        # 2. 相手の勝利を防ぐ手があるか確認
        blocking_move = self._find_blocking_move(game, legal_moves)
        if blocking_move:
            tile_action = self._select_tile_placement(game, blocking_move)
            return (blocking_move, tile_action)

        # 3. 最も前進できる手を選択
        best_move = self._find_best_advancing_move(game, legal_moves)

        # 4. タイル配置を決定
        tile_action = self._select_tile_placement(game, best_move)

        return (best_move, tile_action)

    def _get_all_legal_moves(
        self, game: ContrastGame
    ) -> List[Tuple[int, int, int, int]]:
        """すべての合法な移動を取得"""
        legal_moves = []

        for y in range(game.board.size):
            for x in range(game.board.size):
                piece = game.board.get_piece(x, y)
                if piece and piece.owner == self.player:
                    valid_moves = game.get_valid_moves(x, y)
                    for to_x, to_y in valid_moves:
                        legal_moves.append((x, y, to_x, to_y))

        return legal_moves

    def _find_winning_move(
        self, game: ContrastGame, legal_moves: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """勝利できる手を探す"""
        goal_y = 0 if self.player == Player.PLAYER1 else game.board.size - 1

        for from_x, from_y, to_x, to_y in legal_moves:
            if to_y == goal_y:
                return (from_x, from_y, to_x, to_y)

        return None

    def _find_blocking_move(
        self, game: ContrastGame, legal_moves: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """相手の勝利を妨害する手を探す"""
        opponent = Player.PLAYER2 if self.player == Player.PLAYER1 else Player.PLAYER1
        opponent_goal_y = 0 if opponent == Player.PLAYER1 else game.board.size - 1

        # 相手がゴールに1手で到達できる位置を確認
        dangerous_positions = []

        for y in range(game.board.size):
            for x in range(game.board.size):
                piece = game.board.get_piece(x, y)
                if piece and piece.owner == opponent:
                    # 相手のゴールまでの距離を確認
                    distance = abs(y - opponent_goal_y)
                    if distance == 1:  # 1手でゴールに到達可能
                        dangerous_positions.append((x, y))

        if not dangerous_positions:
            return None

        # 危険な位置の前に黒タイルを置ける手を探す
        for from_x, from_y, to_x, to_y in legal_moves:
            # 相手のゴールへの経路上に移動する手を優先
            for opp_x, opp_y in dangerous_positions:
                if abs(to_x - opp_x) <= 1 and abs(to_y - opp_y) <= 1:
                    return (from_x, from_y, to_x, to_y)

        return None

    def _find_best_advancing_move(
        self, game: ContrastGame, legal_moves: List[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """最も前進できる手を選択"""
        goal_y = 0 if self.player == Player.PLAYER1 else game.board.size - 1

        best_move = None
        best_score = float("-inf")

        for from_x, from_y, to_x, to_y in legal_moves:
            # 前進度を計算（ゴールに近づくほど高得点）
            from_distance = abs(from_y - goal_y)
            to_distance = abs(to_y - goal_y)
            advance = from_distance - to_distance

            # タイルの色による評価
            tile_color = game.board.get_tile_color(to_x, to_y)
            tile_bonus = 0
            if tile_color == TileColor.WHITE:
                tile_bonus = 0
            elif tile_color == TileColor.BLACK:
                tile_bonus = 2  # 黒タイルへの移動は高評価
            else:  # GRAY
                tile_bonus = 1  # グレータイルは中評価

            # 中央に近いほど選択肢が増えるボーナス
            center_x = game.board.size // 2
            center_bonus = -abs(to_x - center_x) * 0.5

            score = advance * 10 + tile_bonus + center_bonus

            if score > best_score:
                best_score = score
                best_move = (from_x, from_y, to_x, to_y)

        return best_move if best_move else legal_moves[0]

    def _select_tile_placement(
        self, game: ContrastGame, move: Tuple[int, int, int, int]
    ) -> Optional[Tuple]:
        """タイル配置を選択"""
        tiles_remaining = game.tiles_remaining[self.player]

        # タイルが残っていなければ配置しない
        if tiles_remaining["black"] <= 0 and tiles_remaining["gray"] <= 0:
            return None

        from_x, from_y, to_x, to_y = move
        goal_y = 0 if self.player == Player.PLAYER1 else game.board.size - 1
        opponent = Player.PLAYER2 if self.player == Player.PLAYER1 else Player.PLAYER1
        opponent_goal_y = 0 if opponent == Player.PLAYER1 else game.board.size - 1

        # タイル配置候補を評価
        best_tile_pos = None
        best_tile_color = None
        best_tile_score = -1

        for y in range(game.board.size):
            for x in range(game.board.size):
                # 配置可能な場所かチェック
                if game.board.get_piece(x, y) is not None:
                    continue
                if game.board.get_tile_color(x, y) != TileColor.WHITE:
                    continue

                # 黒タイルの評価
                if tiles_remaining["black"] > 0:
                    score = self._evaluate_tile_placement(
                        game, x, y, TileColor.BLACK, goal_y, opponent_goal_y
                    )
                    if score > best_tile_score:
                        best_tile_score = score
                        best_tile_pos = (x, y)
                        best_tile_color = TileColor.BLACK

                # グレータイルの評価
                if tiles_remaining["gray"] > 0:
                    score = self._evaluate_tile_placement(
                        game, x, y, TileColor.GRAY, goal_y, opponent_goal_y
                    )
                    if score > best_tile_score:
                        best_tile_score = score
                        best_tile_pos = (x, y)
                        best_tile_color = TileColor.GRAY

        # スコアが低すぎる場合は配置しない（温存）
        if best_tile_score < 5:
            return None

        if best_tile_pos and best_tile_color:
            return (best_tile_color, best_tile_pos[0], best_tile_pos[1])

        return None

    def _evaluate_tile_placement(
        self,
        game: ContrastGame,
        x: int,
        y: int,
        tile_color: TileColor,
        goal_y: int,
        opponent_goal_y: int,
    ) -> float:
        """タイル配置の評価値を計算"""
        score = 0.0

        # 自分のゴールへの経路上にある場合
        my_path_distance = abs(y - goal_y)
        if my_path_distance < game.board.size // 2:
            if tile_color == TileColor.BLACK:
                score += 15  # 自分の経路に黒タイルは高評価
            else:
                score += 8  # グレータイルも有用

        # 相手のゴールへの経路上にある場合
        opp_path_distance = abs(y - opponent_goal_y)
        if opp_path_distance < game.board.size // 2:
            if tile_color == TileColor.BLACK:
                score += 10  # 相手の経路を妨害
            else:
                score += 5

        # 相手のコマの近くに配置する場合
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < game.board.size and 0 <= ny < game.board.size:
                    piece = game.board.get_piece(nx, ny)
                    if piece and piece.owner != self.player:
                        if tile_color == TileColor.BLACK:
                            score += 5  # 相手の選択肢を減らす
                        else:
                            score += 3

        # 自分のコマの前方に配置する場合
        for y_piece in range(game.board.size):
            for x_piece in range(game.board.size):
                piece = game.board.get_piece(x_piece, y_piece)
                if piece and piece.owner == self.player:
                    distance = abs(x_piece - x) + abs(y_piece - y)
                    if distance <= 2:
                        direction = 1 if self.player == Player.PLAYER1 else -1
                        if (y - y_piece) * direction > 0:  # 前方
                            if tile_color == TileColor.BLACK:
                                score += 8
                            else:
                                score += 4

        # ランダム要素を追加（同点の場合の多様性のため）
        score += random.random() * 2

        return score


if __name__ == "__main__":
    # テスト
    print("ルールベースプレイヤーのテスト")
    print("=" * 60)

    game = ContrastGame(board_size=5)
    game.setup_initial_position()

    player1 = RuleBasedPlayer(Player.PLAYER1)
    player2 = RuleBasedPlayer(Player.PLAYER2)

    move_count = 0
    max_moves = 100

    while not game.game_over and move_count < max_moves:
        current_player = player1 if game.current_player == Player.PLAYER1 else player2

        action = current_player.select_action(game)

        if action is None:
            print(f"Player {game.current_player.value} has no valid moves!")
            break

        move_tuple, tile_tuple = action
        from_x, from_y, to_x, to_y = move_tuple

        if tile_tuple:
            place_tile = True
            tile_color, tile_x, tile_y = tile_tuple
            print(
                f"Player {game.current_player.value}: ({from_x},{from_y}) -> ({to_x},{to_y}), "
                f"Tile: {tile_color.value} at ({tile_x},{tile_y})"
            )
        else:
            place_tile = False
            tile_x, tile_y, tile_color = None, None, None
            print(
                f"Player {game.current_player.value}: ({from_x},{from_y}) -> ({to_x},{to_y})"
            )

        game.make_move(
            from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
        )
        move_count += 1

    print("=" * 60)
    if game.game_over:
        print(
            f"Game Over! Winner: Player {game.winner.value if game.winner else 'None'}"
        )
    else:
        print(f"Max moves ({max_moves}) reached")
    print(f"Total moves: {move_count}")
