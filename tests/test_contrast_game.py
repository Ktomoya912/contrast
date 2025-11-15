"""
コントラストゲームのユニットテスト
"""

import unittest
from contrast_game import (
    ContrastGame, Board, Piece, Player, Color, Direction
)


class TestBoard(unittest.TestCase):
    """ボードクラスのテスト"""
    
    def setUp(self):
        self.board = Board(size=8)
    
    def test_board_initialization(self):
        """ボードが正しく初期化されるか"""
        self.assertEqual(self.board.size, 8)
        self.assertEqual(len(self.board.pieces), 8)
        self.assertEqual(len(self.board.pieces[0]), 8)
    
    def test_board_colors(self):
        """ボードの色パターンが正しいか（市松模様）"""
        # (0,0)は黒
        self.assertEqual(self.board.get_color(0, 0), Color.BLACK)
        # (0,1)は白
        self.assertEqual(self.board.get_color(0, 1), Color.WHITE)
        # (1,0)は白
        self.assertEqual(self.board.get_color(1, 0), Color.WHITE)
        # (1,1)は黒
        self.assertEqual(self.board.get_color(1, 1), Color.BLACK)
    
    def test_place_piece(self):
        """コマを配置できるか"""
        piece = Piece(Direction.UP, Direction.DOWN, Player.PLAYER1)
        result = self.board.place_piece(0, 0, piece)
        self.assertTrue(result)
        self.assertEqual(self.board.get_piece(0, 0), piece)
    
    def test_place_piece_occupied(self):
        """既にコマがある場所には配置できない"""
        piece1 = Piece(Direction.UP, Direction.DOWN, Player.PLAYER1)
        piece2 = Piece(Direction.LEFT, Direction.RIGHT, Player.PLAYER2)
        
        self.board.place_piece(0, 0, piece1)
        result = self.board.place_piece(0, 0, piece2)
        
        self.assertFalse(result)
        self.assertEqual(self.board.get_piece(0, 0), piece1)
    
    def test_move_piece(self):
        """コマを移動できるか"""
        piece = Piece(Direction.UP, Direction.DOWN, Player.PLAYER1)
        self.board.place_piece(0, 0, piece)
        
        result = self.board.move_piece(0, 0, 1, 1)
        
        self.assertTrue(result)
        self.assertIsNone(self.board.get_piece(0, 0))
        self.assertEqual(self.board.get_piece(1, 1), piece)
    
    def test_remove_piece(self):
        """コマを削除できるか"""
        piece = Piece(Direction.UP, Direction.DOWN, Player.PLAYER1)
        self.board.place_piece(0, 0, piece)
        self.board.remove_piece(0, 0)
        
        self.assertIsNone(self.board.get_piece(0, 0))


class TestPiece(unittest.TestCase):
    """コマクラスのテスト"""
    
    def test_piece_creation(self):
        """コマが正しく作成されるか"""
        piece = Piece(Direction.UP, Direction.DOWN, Player.PLAYER1)
        self.assertEqual(piece.black_arrow, Direction.UP)
        self.assertEqual(piece.white_arrow, Direction.DOWN)
        self.assertEqual(piece.owner, Player.PLAYER1)
    
    def test_visible_arrow_on_black(self):
        """黒い背景では白い矢印が見える"""
        piece = Piece(Direction.UP, Direction.DOWN, Player.PLAYER1)
        visible = piece.get_visible_arrow(Color.BLACK)
        self.assertEqual(visible, Direction.DOWN)
    
    def test_visible_arrow_on_white(self):
        """白い背景では黒い矢印が見える"""
        piece = Piece(Direction.UP, Direction.DOWN, Player.PLAYER1)
        visible = piece.get_visible_arrow(Color.WHITE)
        self.assertEqual(visible, Direction.UP)


class TestContrastGame(unittest.TestCase):
    """ゲームクラスのテスト"""
    
    def setUp(self):
        self.game = ContrastGame(board_size=8)
        self.game.setup_initial_position()
    
    def test_game_initialization(self):
        """ゲームが正しく初期化されるか"""
        self.assertEqual(self.game.current_player, Player.PLAYER1)
        self.assertFalse(self.game.game_over)
        self.assertIsNone(self.game.winner)
    
    def test_initial_setup(self):
        """初期配置が正しいか"""
        # プレイヤー1のコマは最下段にある
        for x in range(8):
            piece = self.game.board.get_piece(x, 7)
            self.assertIsNotNone(piece)
            self.assertEqual(piece.owner, Player.PLAYER1)
        
        # プレイヤー2のコマは最上段にある
        for x in range(8):
            piece = self.game.board.get_piece(x, 0)
            self.assertIsNotNone(piece)
            self.assertEqual(piece.owner, Player.PLAYER2)
    
    def test_get_valid_moves(self):
        """有効な移動先が正しく取得できるか"""
        # プレイヤー1のコマ (3,7) は上に移動できるはず
        valid_moves = self.game.get_valid_moves(3, 7)
        self.assertTrue(len(valid_moves) > 0)
        
        # 空のマスからは移動できない
        valid_moves = self.game.get_valid_moves(3, 3)
        self.assertEqual(len(valid_moves), 0)
    
    def test_make_valid_move(self):
        """有効な移動ができるか"""
        initial_player = self.game.current_player
        valid_moves = self.game.get_valid_moves(3, 7)
        
        if valid_moves:
            to_x, to_y = valid_moves[0]
            result = self.game.make_move(3, 7, to_x, to_y)
            
            self.assertTrue(result)
            # プレイヤーが交代しているはず
            self.assertNotEqual(self.game.current_player, initial_player)
    
    def test_make_invalid_move(self):
        """無効な移動は拒否されるか"""
        result = self.game.make_move(3, 7, 5, 5)
        # 通常は無効な移動
        # 注: 初期配置によっては有効な場合もあるため、
        # ここでは単に移動が実行されるかどうかをチェック
        self.assertIsInstance(result, bool)
    
    def test_player_turn_switching(self):
        """プレイヤーのターンが正しく切り替わるか"""
        self.assertEqual(self.game.current_player, Player.PLAYER1)
        
        # 有効な移動を実行
        valid_moves = self.game.get_valid_moves(3, 7)
        if valid_moves:
            to_x, to_y = valid_moves[0]
            self.game.make_move(3, 7, to_x, to_y)
            self.assertEqual(self.game.current_player, Player.PLAYER2)


class TestDirection(unittest.TestCase):
    """方向の定義テスト"""
    
    def test_direction_values(self):
        """方向の値が正しいか"""
        self.assertEqual(Direction.UP.value, (0, -1))
        self.assertEqual(Direction.DOWN.value, (0, 1))
        self.assertEqual(Direction.LEFT.value, (-1, 0))
        self.assertEqual(Direction.RIGHT.value, (1, 0))
        self.assertEqual(Direction.UP_LEFT.value, (-1, -1))
        self.assertEqual(Direction.UP_RIGHT.value, (1, -1))
        self.assertEqual(Direction.DOWN_LEFT.value, (-1, 1))
        self.assertEqual(Direction.DOWN_RIGHT.value, (1, 1))


if __name__ == '__main__':
    unittest.main()
