"""
コントラストゲームのルールテスト
公式ルール（https://replaygames.blog/contrast/）に基づいたテスト
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest

from contrast_game import ContrastGame, Piece, Player, TileColor


class TestContrastBasicRules(unittest.TestCase):
    """基本ルールのテスト"""

    def setUp(self):
        """各テストの前に実行"""
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_initial_setup(self):
        """初期配置のテスト"""
        # プレイヤー1は下側（y=4）に5つのコマ
        for x in range(5):
            piece = self.game.board.get_piece(x, 4)
            self.assertIsNotNone(piece)
            self.assertEqual(piece.owner, Player.PLAYER1)

        # プレイヤー2は上側（y=0）に5つのコマ
        for x in range(5):
            piece = self.game.board.get_piece(x, 0)
            self.assertIsNotNone(piece)
            self.assertEqual(piece.owner, Player.PLAYER2)

    def test_initial_tile_count(self):
        """初期タイル数のテスト"""
        # 各プレイヤーは黒タイル3枚、グレータイル1枚を持つ
        self.assertEqual(self.game.tiles_remaining[Player.PLAYER1]["black"], 3)
        self.assertEqual(self.game.tiles_remaining[Player.PLAYER1]["gray"], 1)
        self.assertEqual(self.game.tiles_remaining[Player.PLAYER2]["black"], 3)
        self.assertEqual(self.game.tiles_remaining[Player.PLAYER2]["gray"], 1)

    def test_white_tile_movement(self):
        """白タイル上での移動（前後左右）のテスト"""
        # プレイヤー1のコマが白タイル上から前後左右に移動可能
        valid_moves = self.game.get_valid_moves(2, 4)
        # 左右には自分のコマがいるので移動不可、前方(2,3)のみ
        expected_moves = [(2, 3)]  # 前
        self.assertEqual(set(valid_moves), set(expected_moves))


class TestTilePlacementRules(unittest.TestCase):
    """タイル配置ルールのテスト"""

    def setUp(self):
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_tile_placement_order(self):
        """タイル配置のタイミング：コマ移動→タイル配置の順"""
        # コマを移動してからタイルを配置
        result = self.game.make_move(
            from_x=2,
            from_y=4,
            to_x=2,
            to_y=3,
            place_tile=True,
            tile_x=1,
            tile_y=2,
            tile_color=TileColor.BLACK,
        )
        self.assertTrue(result)

        # タイルが配置されているか確認
        self.assertEqual(self.game.board.get_tile_color(1, 2), TileColor.BLACK)

        # コマが移動しているか確認
        self.assertIsNotNone(self.game.board.get_piece(2, 3))
        self.assertIsNone(self.game.board.get_piece(2, 4))

    def test_cannot_place_tile_on_piece(self):
        """コマがある場所にタイルを置けない"""
        # プレイヤー1のターン：コマを移動
        self.game.make_move(2, 4, 2, 3)

        # プレイヤー2のターン：コマがある場所にタイルを置こうとする
        result = self.game.place_tile(2, 3, TileColor.BLACK)  # コマがいる場所
        self.assertFalse(result)

    def test_cannot_place_tile_on_existing_tile(self):
        """既にタイルがある場所にタイルを置けない"""
        # 黒タイルを配置
        self.game.place_tile(2, 2, TileColor.BLACK)

        # 同じ場所にグレータイルを置こうとする
        result = self.game.place_tile(2, 2, TileColor.GRAY)
        self.assertFalse(result)

    def test_tile_placement_is_optional(self):
        """タイル配置は任意（置かなくても良い）"""
        # タイルを置かずにコマだけ移動
        result = self.game.make_move(2, 4, 2, 3, place_tile=False)
        self.assertTrue(result)

        # タイル残数が減っていないことを確認
        self.assertEqual(self.game.tiles_remaining[Player.PLAYER1]["black"], 3)


class TestMovementRules(unittest.TestCase):
    """コマの移動ルールのテスト"""

    def setUp(self):
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_black_tile_diagonal_movement(self):
        """黒タイル上では斜め移動のみ可能"""
        # 黒タイルを配置
        self.game.board.set_tile_color(2, 3, TileColor.BLACK)

        # プレイヤー1のコマを黒タイル上に移動
        self.game.make_move(2, 4, 2, 3)

        # プレイヤー2のターンをスキップするためダミー移動
        self.game.make_move(0, 0, 0, 1)

        # 黒タイル上のコマは斜めにしか移動できない
        valid_moves = self.game.get_valid_moves(2, 3)
        # 斜め方向のみ（左上、右上）、下は自分のコマがいるので除外
        expected_diagonals = [(1, 2), (3, 2)]  # 左上、右上
        self.assertEqual(set(valid_moves), set(expected_diagonals))

    def test_gray_tile_all_direction_movement(self):
        """グレータイル上では全方向移動可能"""
        # グレータイルを配置
        self.game.board.set_tile_color(2, 3, TileColor.GRAY)

        # プレイヤー1のコマをグレータイル上に移動
        self.game.make_move(2, 4, 2, 3)

        # プレイヤー2のターンをスキップ
        self.game.make_move(0, 0, 0, 1)

        # グレータイル上のコマは全方向に移動可能
        valid_moves = self.game.get_valid_moves(2, 3)
        # 自分のコマがいる下方向を除外
        expected_all = [
            (1, 2),
            (2, 2),
            (3, 2),  # 上方向
            (1, 3),
            (3, 3),  # 左右
            (2, 4),  # 下（中央のみ、左右は自分のコマ）
        ]
        self.assertEqual(set(valid_moves), set(expected_all))

    def test_cannot_move_to_opponent_piece(self):
        """相手のコマがいる場所には移動できない"""
        # コマを近づける設定
        self.game.board.remove_piece(2, 0)
        self.game.board.place_piece(2, 2, Piece(Player.PLAYER2))

        # プレイヤー1が相手のコマがいる場所に移動しようとする
        valid_moves = self.game.get_valid_moves(2, 4)

        # (2,2)には相手のコマがいるので移動できない
        self.assertNotIn((2, 2), valid_moves)

    def test_can_jump_over_own_pieces(self):
        """自分のコマは飛び越えられる"""
        # プレイヤー1のコマを一直線に並べて飛び越えテスト
        # (2,4) -> (2,3)に移動
        self.game.make_move(2, 4, 2, 3)
        self.game.make_move(0, 0, 0, 1)  # P2のターン

        # (3,4)から上に移動して、(2,3)の右に配置
        self.game.make_move(3, 4, 3, 3)
        self.game.make_move(0, 1, 0, 2)  # P2のターン

        # (3,3)から左に移動して(2,3)の隣に
        self.game.make_move(3, 3, 2, 3)
        # これは失敗する（(2,3)には既にコマがいる）
        # 代わりに別の手順で配置

        # やり直し：縦に並べる
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

        # (2,4) -> (2,3)
        self.game.make_move(2, 4, 2, 3)
        self.game.make_move(0, 0, 0, 1)

        # (2,3) -> (2,2)
        self.game.make_move(2, 3, 2, 2)
        self.game.make_move(0, 1, 0, 2)

        # (3,4) -> (3,3)
        self.game.make_move(3, 4, 3, 3)
        self.game.make_move(0, 2, 0, 3)

        # (3,3) -> (2,3) でP1のコマを(2,3)に配置
        self.game.make_move(3, 3, 2, 3)
        self.game.make_move(1, 0, 1, 1)

        # 今の状態: (2,4)[空] (2,3)[P1] (2,2)[P1]
        # (2,4)に新しいコマを配置
        self.game.make_move(1, 4, 2, 4)
        self.game.make_move(1, 1, 1, 2)

        # 状態: (2,4)[P1] (2,3)[P1] (2,2)[P1] が縦に並んでいる
        # (2,4)から上方向に見ると、(2,3)と(2,2)を飛び越えて(2,1)に移動可能
        valid_moves = self.game.get_valid_moves(2, 4)

        # (2,2)を飛び越えた先の(2,1)が有効な移動先
        self.assertIn((2, 1), valid_moves)

        # 実際に飛び越えて移動
        result = self.game.make_move(2, 4, 2, 1)
        self.assertTrue(result)

        # コマが正しく移動したことを確認
        self.assertIsNotNone(self.game.board.get_piece(2, 1))
        self.assertIsNone(self.game.board.get_piece(2, 4))

    def test_can_jump_over_multiple_own_pieces(self):
        """複数の自分のコマを飛び越えられる"""
        # Player 1のコマを一直線に複数並べる
        self.game.make_move(2, 4, 2, 3)  # P1: (2,3)
        self.game.make_move(0, 0, 0, 1)  # P2

        self.game.make_move(2, 3, 2, 2)  # P1: (2,2)
        self.game.make_move(0, 1, 0, 2)  # P2

        self.game.make_move(1, 4, 1, 3)  # P1
        self.game.make_move(0, 2, 0, 3)  # P2

        self.game.make_move(1, 3, 2, 3)  # P1: (2,3)に配置
        self.game.make_move(1, 0, 1, 1)  # P2

        self.game.make_move(0, 4, 1, 4)  # P1
        self.game.make_move(1, 1, 1, 2)  # P2

        self.game.make_move(1, 4, 2, 4)  # P1: (2,4)に配置
        self.game.make_move(1, 2, 1, 3)  # P2

        # 今の状態: (2,4)[P1] (2,3)[P1] (2,2)[P1] が縦に3つ並んでいる
        # (2,4)から上方向に3つ（実際は2つ）のコマを飛び越えて(2,1)まで移動可能
        valid_moves = self.game.get_valid_moves(2, 4)

        # 飛び越えた先の(2,1)が有効な移動先に含まれる
        self.assertIn((2, 1), valid_moves)

    def test_cannot_jump_over_opponent_pieces(self):
        """相手のコマは飛び越えられない"""
        # プレイヤー1とプレイヤー2のコマを近づける
        self.game.make_move(2, 4, 2, 3)  # P1
        self.game.make_move(2, 0, 2, 1)  # P2

        self.game.make_move(2, 3, 2, 2)  # P1: (2,2)に移動
        self.game.make_move(3, 0, 3, 1)  # P2

        # 今の状態: (2,2)[P1] (2,1)[P2]
        # P1が(2,2)から上(2,1)方向を見ると、相手のコマがある

        # P1のターン: (2,2)のコマの移動先を確認
        valid_moves = self.game.get_valid_moves(2, 2)

        # (2,1)には相手のコマがいるので移動できない
        self.assertNotIn((2, 1), valid_moves)
        # (2,0)も相手のコマを飛び越えることになるので移動できない
        self.assertNotIn((2, 0), valid_moves)


class TestWinConditions(unittest.TestCase):
    """勝利条件のテスト"""

    def setUp(self):
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_player1_reaches_opponent_territory(self):
        """プレイヤー1が相手陣地（y=0）に到達して勝利"""
        # プレイヤー1のコマを相手陣地まで移動させる
        self.game.board.remove_piece(2, 4)
        self.game.board.remove_piece(2, 0)
        self.game.board.place_piece(2, 1, Piece(Player.PLAYER1))

        # 最後の1マスを移動
        self.game.make_move(2, 1, 2, 0)

        # ゲームが終了し、プレイヤー1が勝者
        self.assertTrue(self.game.game_over)
        self.assertEqual(self.game.winner, Player.PLAYER1)

    def test_player2_reaches_opponent_territory(self):
        """プレイヤー2が相手陣地（y=4）に到達して勝利"""
        # プレイヤー2のコマを相手陣地まで移動させる
        self.game.board.remove_piece(2, 0)
        self.game.board.remove_piece(2, 4)
        self.game.board.place_piece(2, 3, Piece(Player.PLAYER2))

        # プレイヤー1のターンをスキップ
        self.game.make_move(0, 4, 0, 3)

        # プレイヤー2が最後の1マスを移動
        self.game.make_move(2, 3, 2, 4)

        # ゲームが終了し、プレイヤー2が勝者
        self.assertTrue(self.game.game_over)
        self.assertEqual(self.game.winner, Player.PLAYER2)


class TestForbiddenActions(unittest.TestCase):
    """禁止行動のテスト"""

    def setUp(self):
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_cannot_capture_opponent_piece(self):
        """相手のコマを取る行為は禁止"""
        # コマを近づける
        self.game.board.remove_piece(2, 0)
        self.game.board.place_piece(2, 3, Piece(Player.PLAYER2))

        # プレイヤー1が相手のコマの位置に移動しようとする
        result = self.game.make_move(2, 4, 2, 3)

        # 移動は失敗する
        self.assertFalse(result)

    def test_lose_when_no_valid_moves(self):
        """動けない状態になったら負け"""
        # プレイヤー1のコマをすべて削除して1つだけ残す
        for x in range(5):
            self.game.board.remove_piece(x, 4)

        # (2,2)に孤立させる
        self.game.board.place_piece(2, 2, Piece(Player.PLAYER1))

        # 周りを相手のコマで囲む
        self.game.board.place_piece(1, 2, Piece(Player.PLAYER2))
        self.game.board.place_piece(3, 2, Piece(Player.PLAYER2))
        self.game.board.place_piece(2, 1, Piece(Player.PLAYER2))
        self.game.board.place_piece(2, 3, Piece(Player.PLAYER2))

        # プレイヤー1のターンに設定
        self.game.current_player = Player.PLAYER1

        # 動けないことを確認
        self.game._check_no_valid_moves()

        # ゲームが終了し、プレイヤー2が勝者
        self.assertTrue(self.game.game_over)
        self.assertEqual(self.game.winner, Player.PLAYER2)


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""

    def setUp(self):
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_board_boundary(self):
        """盤面の境界チェック"""
        # 端のコマの移動可能範囲
        valid_moves = self.game.get_valid_moves(0, 4)

        # 盤面外には移動できない
        self.assertNotIn((-1, 4), valid_moves)
        self.assertNotIn((0, 5), valid_moves)

    def test_tile_count_decreases(self):
        """タイル配置でタイル数が減る"""
        initial_black = self.game.tiles_remaining[Player.PLAYER1]["black"]

        # 黒タイルを配置
        self.game.make_move(2, 4, 2, 3, True, 1, 2, TileColor.BLACK)

        # タイル数が1減る
        self.assertEqual(
            self.game.tiles_remaining[Player.PLAYER1]["black"], initial_black - 1
        )

    def test_cannot_place_tile_when_none_remaining(self):
        """タイルがなくなったら配置できない"""
        # すべての黒タイルを使い切る
        self.game.tiles_remaining[Player.PLAYER1]["black"] = 0

        # 黒タイルを配置しようとする
        result = self.game.place_tile(2, 2, TileColor.BLACK)

        # 配置できない
        self.assertFalse(result)

    def test_player_alternation(self):
        """プレイヤーが交互にターンを進める"""
        self.assertEqual(self.game.current_player, Player.PLAYER1)

        # プレイヤー1が移動
        self.game.make_move(2, 4, 2, 3)
        self.assertEqual(self.game.current_player, Player.PLAYER2)

        # プレイヤー2が移動
        self.game.make_move(2, 0, 2, 1)
        self.assertEqual(self.game.current_player, Player.PLAYER1)


class TestComplexScenarios(unittest.TestCase):
    """複雑なシナリオのテスト"""

    def setUp(self):
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_tile_strategy_blocking(self):
        """タイルを使った妨害戦略"""
        # プレイヤー1が前進
        self.game.make_move(2, 4, 2, 3)

        # プレイヤー2が黒タイルを配置して進路を妨害
        # (2,2)に黒タイルを置くと、(2,3)から(2,2)への縦移動ができなくなる
        self.game.make_move(2, 0, 2, 1, True, 2, 2, TileColor.BLACK)

        # プレイヤー1のコマ(2,3)は白タイル上なので縦横移動
        # しかし(2,2)は黒タイルなので、縦移動先として(2,2)は有効
        # （タイルの色は移動元で判定される）
        valid_moves = self.game.get_valid_moves(2, 3)
        self.assertIn((2, 2), valid_moves)  # 白タイル上からは縦移動可能

    def test_multiple_tiles_placement(self):
        """複数のタイルを配置するゲーム進行"""
        moves = [
            (2, 4, 2, 3, True, 1, 2, TileColor.BLACK),  # P1
            (2, 0, 2, 1, True, 3, 2, TileColor.BLACK),  # P2
            (2, 3, 2, 2, True, 2, 3, TileColor.GRAY),  # P1
            (
                2,
                1,
                2,
                2,
                False,
                None,
                None,
                None,
            ),  # P2（P1のコマと同じ位置には移動不可）
        ]

        for move in moves[:3]:
            result = self.game.make_move(*move)
            self.assertTrue(result)

        # 4番目の移動は失敗する（同じ位置に移動不可）
        result = self.game.make_move(*moves[3])
        self.assertFalse(result)


class TestJumpingMechanics(unittest.TestCase):
    """飛び越え機能の詳細テスト"""

    def setUp(self):
        self.game = ContrastGame(board_size=5)
        self.game.setup_initial_position()

    def test_jump_stops_at_first_empty_space(self):
        """飛び越え後、最初の空マスで停止できる"""
        # (2,4)のコマの上に(2,3)のコマを配置
        self.game.make_move(1, 4, 1, 3)  # P1
        self.game.make_move(0, 0, 0, 1)  # P2

        # (2,4)から上に移動：(2,3)は自分のコマがいないので通常移動
        valid_moves = self.game.get_valid_moves(2, 4)
        self.assertIn((2, 3), valid_moves)

    def test_jump_with_black_tile_diagonal(self):
        """黒タイル上での斜め飛び越え"""
        # 黒タイルを(3,4)に配置
        self.game.board.set_tile_color(3, 4, TileColor.BLACK)

        # (3,4)のコマを斜め上に配置するため、まず(4,3)に移動
        # その前に(4,4)を移動
        self.game.make_move(4, 4, 4, 3)  # P1
        self.game.make_move(0, 0, 0, 1)  # P2

        # (3,4)は黒タイルなので斜め移動のみ
        # (4,3)に自分のコマがいるので、飛び越えて(5,2)...は盤面外
        valid_moves = self.game.get_valid_moves(3, 4)

        # 右上斜めに(4,3)を飛び越える先は盤面外なので移動不可
        self.assertNotIn((5, 2), valid_moves)

        # 左上斜めは(2,3)が空なので通常移動
        self.assertIn((2, 3), valid_moves)

    def test_jump_chain_stops_at_opponent(self):
        """飛び越え中に相手のコマに遭遇したら停止"""
        # 複雑な配置：自分→自分→相手の順
        self.game.make_move(2, 4, 2, 3)  # P1: (2,3)
        self.game.make_move(2, 0, 2, 1)  # P2: (2,1)

        self.game.make_move(1, 4, 1, 3)  # P1
        self.game.make_move(3, 0, 3, 1)  # P2

        self.game.make_move(1, 3, 2, 2)  # P1: (2,2)
        self.game.make_move(3, 1, 3, 2)  # P2

        # 状態: (2,3)[P1] (2,2)[P1] (2,1)[P2]
        # (2,4)から上方向: 自分→自分→相手
        # (3,4)から左上を見る
        self.game.make_move(3, 4, 3, 3)  # P1
        self.game.make_move(3, 2, 3, 3)  # P2は同じ場所に移動不可、別の手
        self.game.make_move(4, 0, 4, 1)  # P2

        # (3,3)から(2,2)方向：(2,2)は自分のコマ
        valid_moves = self.game.get_valid_moves(3, 3)
        # (2,2)を飛び越えて(2,1)...は相手のコマなので不可
        # 実際の移動先候補を確認
        # 斜め左上(2,2)を飛び越える方向では相手(2,1)で停止
        self.assertNotIn((2, 1), valid_moves)

    def test_no_jump_without_own_piece(self):
        """自分のコマがない場合は通常の1マス移動のみ"""
        # 初期状態で(2,4)の前(2,3)は空
        valid_moves = self.game.get_valid_moves(2, 4)

        # (2,3)には移動できるが(2,2)には移動できない（飛び越えるコマがない）
        self.assertIn((2, 3), valid_moves)
        self.assertNotIn((2, 2), valid_moves)

    def test_jump_ends_at_board_edge(self):
        """飛び越えても盤面の端で停止"""
        # (0,4)のコマを使用
        # 左方向には盤面外なので移動不可
        valid_moves = self.game.get_valid_moves(0, 4)

        # 左(-1,4)は盤面外
        self.assertNotIn((-1, 4), valid_moves)

        # 上(0,3)は空なので移動可能
        self.assertIn((0, 3), valid_moves)


if __name__ == "__main__":
    # テストを実行
    unittest.main(verbosity=2)
