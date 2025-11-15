"""
コントラスト - 2人対戦ボードゲーム
5x5の盤面で、タイルの色によって移動方向が変化する戦略的なゲーム
"""

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class TileColor(Enum):
    """タイルの色"""

    WHITE = "white"  # 白：縦横移動
    BLACK = "black"  # 黒：斜め移動
    GRAY = "gray"  # グレー：全方向移動


class Player(Enum):
    """プレイヤー"""

    PLAYER1 = 1
    PLAYER2 = 2


class Piece:
    """コマ"""

    def __init__(self, owner: Player):
        """
        Args:
            owner: コマの所有者
        """
        self.owner = owner

    def __repr__(self):
        return f"Piece(P{self.owner.value})"


class Board:
    """ゲームボード（5x5）- NumPy最適化版"""

    def __init__(self, size: int = 5):
        """
        Args:
            size: ボードのサイズ（size x size）
        """
        self.size = size
        # タイルをNumPy配列で管理（0=WHITE, 1=BLACK, 2=GRAY）
        self.tiles = np.zeros((size, size), dtype=np.int8)

        # コマをNumPy配列で管理（0=なし, 1=PLAYER1, 2=PLAYER2）
        self.pieces = np.zeros((size, size), dtype=np.int8)

        # タイルとプレイヤーのマッピング
        self._tile_map = {0: TileColor.WHITE, 1: TileColor.BLACK, 2: TileColor.GRAY}
        self._tile_rmap = {TileColor.WHITE: 0, TileColor.BLACK: 1, TileColor.GRAY: 2}
        self._player_map = {0: None, 1: Player.PLAYER1, 2: Player.PLAYER2}
        self._player_rmap = {None: 0, Player.PLAYER1: 1, Player.PLAYER2: 2}

    def get_tile_color(self, x: int, y: int) -> TileColor:
        """指定位置のタイル色を取得"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self._tile_map[self.tiles[y, x]]
        raise ValueError(f"Invalid position: ({x}, {y})")

    def set_tile_color(self, x: int, y: int, color: TileColor) -> bool:
        """
        タイルの色を設定する

        Args:
            x, y: 位置
            color: タイルの色

        Returns:
            設定に成功したかどうか
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            self.tiles[y, x] = self._tile_rmap[color]
            return True
        return False

    def place_piece(self, x: int, y: int, piece: Piece) -> bool:
        """
        コマを配置する

        Args:
            x, y: 配置位置
            piece: 配置するコマ

        Returns:
            配置に成功したかどうか
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            if self.pieces[y, x] == 0:
                self.pieces[y, x] = self._player_rmap[piece.owner]
                return True
        return False

    def get_piece(self, x: int, y: int) -> Optional[Piece]:
        """指定位置のコマを取得"""
        if 0 <= x < self.size and 0 <= y < self.size:
            player = self._player_map[self.pieces[y, x]]
            return Piece(player) if player else None
        return None

    def move_piece(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """
        コマを移動する

        Args:
            from_x, from_y: 移動元の位置
            to_x, to_y: 移動先の位置

        Returns:
            移動に成功したかどうか
        """
        if self.pieces[from_y, from_x] == 0:
            return False

        if 0 <= to_x < self.size and 0 <= to_y < self.size:
            self.pieces[to_y, to_x] = self.pieces[from_y, from_x]
            self.pieces[from_y, from_x] = 0
            return True
        return False

    def remove_piece(self, x: int, y: int):
        """コマを盤から取り除く"""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.pieces[y, x] = 0

    def display(self):
        """ボードの状態を表示（NumPy最適化版）"""
        print("  ", end="")
        for i in range(self.size):
            print(f" {i} ", end="")
        print()

        # タイルマーカー
        tile_markers = {0: "□", 1: "■", 2: "▦"}

        for y in range(self.size):
            print(f"{y} ", end="")
            for x in range(self.size):
                piece_id = self.pieces[y, x]
                tile_id = self.tiles[y, x]
                tile_marker = tile_markers[tile_id]

                if piece_id == 0:
                    # コマなし：タイルのみ表示
                    print(f"[{tile_marker}]", end="")
                else:
                    # コマあり：プレイヤー番号 + タイル
                    print(f"[{piece_id}{tile_marker}]", end="")
            print()


class ContrastGame:
    """コントラストゲームのメインクラス"""

    def __init__(self, board_size: int = 5):
        """
        Args:
            board_size: ボードのサイズ
        """
        self.board = Board(board_size)
        self.current_player = Player.PLAYER1
        self.game_over = False
        self.winner = None
        self.move_history = []

        # プレイヤーごとのタイル所持数
        self.tiles_remaining = {
            Player.PLAYER1: {"black": 3, "gray": 1},
            Player.PLAYER2: {"black": 3, "gray": 1},
        }

        # 配置されたタイルの記録（座標とプレイヤー）
        self.placed_tiles: List[Tuple[int, int, Player, TileColor]] = []

    def setup_initial_position(self):
        """初期配置を設定（各プレイヤー5つのコマ）"""
        # プレイヤー1の初期配置（下側・y=4）
        for x in range(self.board.size):
            piece = Piece(owner=Player.PLAYER1)
            self.board.place_piece(x, 4, piece)

        # プレイヤー2の初期配置（上側・y=0）
        for x in range(self.board.size):
            piece = Piece(owner=Player.PLAYER2)
            self.board.place_piece(x, 0, piece)

    def get_valid_moves(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        指定位置のコマが移動可能なマスのリストを取得（NumPy最適化版）
        ルール：自分のコマは飛び越えられる（複数でも可）、相手のコマは飛び越えられない

        Args:
            x, y: コマの位置

        Returns:
            移動可能な位置のリスト [(x, y), ...]
        """
        # 所有者チェック（NumPy配列で高速化）
        player_id = self.board._player_rmap[self.current_player]
        if self.board.pieces[y, x] != player_id:
            return []

        # 現在のマスのタイル色を取得
        tile_id = self.board.tiles[y, x]

        # 方向を決定
        if tile_id == 0:  # WHITE
            directions = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])
        elif tile_id == 1:  # BLACK
            directions = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1)])
        else:  # GRAY
            directions = np.array(
                [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
            )

        valid_moves = []

        for dx, dy in directions:
            # その方向に進む（自分のコマを飛び越える）
            current_x, current_y = x, y

            # まず自分のコマを連続して飛び越える
            while True:
                next_x = current_x + dx
                next_y = current_y + dy

                # 盤面外チェック
                if not (
                    0 <= next_x < self.board.size and 0 <= next_y < self.board.size
                ):
                    break

                target_piece_id = self.board.pieces[next_y, next_x]

                if target_piece_id == 0:
                    # 空マスに到達：ここが移動先候補
                    valid_moves.append((next_x, next_y))
                    break
                elif target_piece_id == player_id:
                    # 自分のコマは飛び越える
                    current_x, current_y = next_x, next_y
                    continue
                else:
                    # 相手のコマで停止（飛び越えられない、移動もできない）
                    break

        return valid_moves

    def place_tile(self, x: int, y: int, tile_color: TileColor) -> bool:
        """
        タイルを配置する
        ルール：空いているマスにのみ配置可能（コマやタイルがあるマスには置けない）

        Args:
            x, y: 配置位置
            tile_color: タイルの色

        Returns:
            配置に成功したかどうか
        """
        # 範囲チェック
        if not (0 <= x < self.board.size and 0 <= y < self.board.size):
            return False

        # すでにコマが置かれている場所には配置できない
        if self.board.get_piece(x, y) is not None:
            return False

        # すでにタイルが置かれている場所（白以外）には配置できない
        if self.board.get_tile_color(x, y) != TileColor.WHITE:
            return False

        # タイルが残っているか確認
        tile_type = "black" if tile_color == TileColor.BLACK else "gray"
        if self.tiles_remaining[self.current_player][tile_type] <= 0:
            return False

        # タイルを配置
        if self.board.set_tile_color(x, y, tile_color):
            self.tiles_remaining[self.current_player][tile_type] -= 1
            self.placed_tiles.append((x, y, self.current_player, tile_color))
            return True

        return False

    def make_move(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        place_tile: bool = False,
        tile_x: int = None,
        tile_y: int = None,
        tile_color: TileColor = None,
    ) -> bool:
        """
        コマを移動する（オプションでタイル配置も可能）
        ルール：コマを移動してから、タイルを配置する順番

        Args:
            from_x, from_y: 移動元の位置
            to_x, to_y: 移動先の位置
            place_tile: タイルを配置するかどうか
            tile_x, tile_y: タイル配置位置
            tile_color: タイルの色

        Returns:
            移動が有効だったかどうか
        """
        # 移動が有効かチェック
        valid_moves = self.get_valid_moves(from_x, from_y)
        if (to_x, to_y) not in valid_moves:
            return False

        # 移動先に相手のコマがないことを確認（念のため再チェック）
        target_piece = self.board.get_piece(to_x, to_y)
        if target_piece is not None:
            return False

        # コマを移動
        if not self.board.move_piece(from_x, from_y, to_x, to_y):
            return False

        # コマ移動後にタイルを配置する（任意）
        tile_placed_info = None
        if (
            place_tile
            and tile_x is not None
            and tile_y is not None
            and tile_color is not None
        ):
            if self.place_tile(tile_x, tile_y, tile_color):
                tile_placed_info = (tile_x, tile_y, tile_color)
            else:
                # タイル配置に失敗してもコマの移動は成功しているので続行
                pass

        # 履歴に記録
        self.move_history.append(
            {
                "player": self.current_player,
                "from": (from_x, from_y),
                "to": (to_x, to_y),
                "tile_placed": tile_placed_info,
            }
        )

        # 勝利条件をチェック
        self._check_win_condition()

        # プレイヤー交代
        if not self.game_over:
            self.current_player = (
                Player.PLAYER2
                if self.current_player == Player.PLAYER1
                else Player.PLAYER1
            )

            # 交代後のプレイヤーが動けるかチェック
            self._check_no_valid_moves()

        return True

    def _check_win_condition(self):
        """勝利条件をチェック：相手の陣地に到達（NumPy最適化版）"""
        # プレイヤー1が相手の陣地（y=0）に到達
        player1_id = self.board._player_rmap[Player.PLAYER1]
        if np.any(self.board.pieces[0, :] == player1_id):
            self.game_over = True
            self.winner = Player.PLAYER1
            return

        # プレイヤー2が相手の陣地（y=4）に到達
        player2_id = self.board._player_rmap[Player.PLAYER2]
        if np.any(self.board.pieces[self.board.size - 1, :] == player2_id):
            self.game_over = True
            self.winner = Player.PLAYER2
            return

    def _check_no_valid_moves(self):
        """
        現在のプレイヤーが動けない状態かチェック（NumPy最適化版）
        ルール：自分のコマが動かせない場合は負け
        """
        if self.game_over:
            return

        # 現在のプレイヤーのすべてのコマについて、合法手があるかチェック
        player_id = self.board._player_rmap[self.current_player]
        # NumPy配列で該当プレイヤーのコマ位置を取得
        player_positions = np.argwhere(self.board.pieces == player_id)

        for pos in player_positions:
            y, x = pos
            valid_moves = self.get_valid_moves(x, y)
            if len(valid_moves) > 0:
                # 動ける手が1つでもあれば続行
                return

        # どのコマも動けない場合は負け
        self.game_over = True
        self.winner = (
            Player.PLAYER2 if self.current_player == Player.PLAYER1 else Player.PLAYER1
        )

    def display_game_state(self):
        """ゲーム状態を表示"""
        print("=" * 40)
        print(f"コントラスト - 現在のプレイヤー: Player {self.current_player.value}")
        print("=" * 40)

        # タイル残数を表示
        p1_tiles = self.tiles_remaining[Player.PLAYER1]
        p2_tiles = self.tiles_remaining[Player.PLAYER2]
        print(
            f"Player 1 タイル残数: 黒■×{p1_tiles['black']} グレー▦×{p1_tiles['gray']}"
        )
        print(
            f"Player 2 タイル残数: 黒■×{p2_tiles['black']} グレー▦×{p2_tiles['gray']}"
        )
        print()

        self.board.display()

        if self.game_over:
            print(f"ゲーム終了！勝者: Player {self.winner.value}")
        print()


def main():
    """ゲームのデモンストレーション"""
    print("コントラスト - ボードゲームシミュレーション")
    print("=" * 50)

    # ゲームを初期化（5x5ボード）
    game = ContrastGame(board_size=5)
    game.setup_initial_position()

    # 初期状態を表示
    game.display_game_state()

    # デモの移動
    print("デモ: いくつかの移動を実行します...\n")

    # Player 1がタイルを配置してから移動
    print("移動 1: Player 1 が黒タイルを(2,3)に配置し、(2,4)から(2,3)へ移動")
    if game.place_tile(2, 3, TileColor.BLACK):
        print("  黒タイル配置成功")
    game.display_game_state()

    moves = [
        # (from_x, from_y, to_x, to_y)
        (2, 4, 2, 3),  # Player 1（黒タイル上なので次は斜めにしか動けない）
        (2, 0, 2, 1),  # Player 2
    ]

    for i, (fx, fy, tx, ty) in enumerate(moves, 2):
        if game.game_over:
            break

        print(
            f"移動 {i}: Player {game.current_player.value} が ({fx}, {fy}) から ({tx}, {ty}) へ移動"
        )

        valid_moves = game.get_valid_moves(fx, fy)
        print(f"有効な移動先: {valid_moves}")

        if game.make_move(fx, fy, tx, ty):
            game.display_game_state()
        else:
            print("無効な移動です！")

    # ゲームの説明
    print("" + "=" * 50)
    print("ゲームのルール:")
    print("=" * 50)
    print("1. 5×5の盤面で各プレイヤー5つのコマを持つ")
    print("2. 通常は白いタイル(□)で、縦横1マスに移動可能")
    print("3. 各プレイヤーは特殊タイルを持っている:")
    print("   - 黒タイル(■) × 3: コマが斜め1マスのみ移動可能")
    print("   - グレータイル(▦) × 1: コマが8方向すべてに移動可能")
    print("4. タイルは一度配置すると移動できない")
    print("5. 勝利条件: 相手の陣地（最初の列）に到達する")
    print("=" * 50)


if __name__ == "__main__":
    main()
