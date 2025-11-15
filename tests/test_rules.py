"""
ルール修正の確認テスト
"""

from contrast_game import ContrastGame, Player

def test_cannot_capture():
    """相手のコマがいる場所には移動できないことを確認"""
    game = ContrastGame()
    game.setup_initial_position()
    
    print("=== 初期状態 ===")
    game.board.display()
    
    # Player 1のコマ(2, 4)を(2, 3)に移動
    print("\n=== Player 1: (2,4) → (2,3) ===")
    result = game.make_move(2, 4, 2, 3)
    print(f"移動結果: {result}")
    game.board.display()
    
    # Player 2のコマ(2, 0)を(2, 1)に移動
    print("\n=== Player 2: (2,0) → (2,1) ===")
    result = game.make_move(2, 0, 2, 1)
    print(f"移動結果: {result}")
    game.board.display()
    
    # Player 1が(2, 3)から(2, 2)に移動しようとする（空マスなので可能）
    print("\n=== Player 1: (2,3) → (2,2) (空マス) ===")
    result = game.make_move(2, 3, 2, 2)
    print(f"移動結果: {result}")
    game.board.display()
    
    # Player 2が(2, 1)から(2, 2)に移動しようとする（P1のコマがいるので不可能）
    print("\n=== Player 2: (2,1) → (2,2) (P1のコマあり) - 失敗すべき ===")
    result = game.make_move(2, 1, 2, 2)
    print(f"移動結果: {result} (Falseであるべき)")
    game.board.display()
    
    # valid_movesで確認
    print("\n=== Player 2の(2,1)からの有効手 ===")
    valid_moves = game.get_valid_moves(2, 1)
    print(f"有効な移動先: {valid_moves}")
    print(f"(2,2)が含まれていない: {(2, 2) not in valid_moves}")
    
    return (2, 2) not in valid_moves and not result

if __name__ == "__main__":
    print("コントラストゲーム - ルール確認テスト")
    print("="*60)
    
    success = test_cannot_capture()
    
    print("\n" + "="*60)
    if success:
        print("✓ テスト成功: 相手のコマがいる場所には移動できません")
    else:
        print("✗ テスト失敗: ルールに問題があります")
