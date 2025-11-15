"""
コントラスト - インタラクティブ版
プレイヤーが対戦できるバージョン
"""

from contrast_game import ContrastGame, Player, TileColor


def get_player_input(prompt: str) -> int:
    """プレイヤーからの数値入力を取得"""
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("数値を入力してください。")


def get_yes_no_input(prompt: str) -> bool:
    """Yes/No入力を取得"""
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes', 'はい', 'h']:
            return True
        elif response in ['n', 'no', 'いいえ', 'i']:
            return False
        else:
            print("y/n で答えてください。")


def play_interactive_game():
    """インタラクティブなゲームプレイ"""
    print("="*60)
    print("コントラスト - 2人対戦ボードゲーム")
    print("="*60)
    print()
    print("ルール:")
    print("- 5×5の盤面で各プレイヤー5つのコマを持ちます")
    print("- 通常は白いタイル(□)で、縦横1マスに移動可能")
    print("- 各プレイヤーは特殊タイルを持っています:")
    print("  * 黒タイル(■) × 3: コマが斜め1マスのみ移動可能")
    print("  * グレータイル(▦) × 1: コマが8方向すべてに移動可能")
    print("- タイルは一度配置すると移動できません")
    print("- 相手の陣地（最初の列）に到達すると勝利です")
    print()
    
    # ボードサイズ
    board_size = 5
    print(f"ボードサイズ: {board_size}x{board_size}")
    
    # ゲームを初期化
    game = ContrastGame(board_size=board_size)
    game.setup_initial_position()
    
    # ゲームループ
    while not game.game_over:
        # 現在の状態を表示
        game.display_game_state()
        
        print(f"\nPlayer {game.current_player.value} のターン")
        
        # タイルを配置するか確認
        place_tile = False
        tile_x = None
        tile_y = None
        tile_color = None
        
        tiles = game.tiles_remaining[game.current_player]
        if tiles['black'] > 0 or tiles['gray'] > 0:
            if get_yes_no_input("タイルを配置しますか？ (y/n): "):
                place_tile = True
                
                # タイルの種類を選択
                while True:
                    print(f"残りタイル - 黒(■): {tiles['black']}, グレー(▦): {tiles['gray']}")
                    tile_type = input("配置するタイルの種類 (black/gray): ").strip().lower()
                    
                    if tile_type in ['black', 'b', '黒', 'くろ'] and tiles['black'] > 0:
                        tile_color = TileColor.BLACK
                        break
                    elif tile_type in ['gray', 'grey', 'g', 'グレー', 'ぐれー'] and tiles['gray'] > 0:
                        tile_color = TileColor.GRAY
                        break
                    else:
                        print("無効な入力またはタイルが残っていません。")
                
                # タイルの配置位置を選択
                print("タイルを配置する位置を選択してください")
                tile_x = get_player_input(f"  X座標 (0-{board_size - 1}): ")
                tile_y = get_player_input(f"  Y座標 (0-{board_size - 1}): ")
        
        # 移動させるコマを選択
        print("\n移動させるコマを選択してください")
        from_x = get_player_input(f"  X座標 (0-{board_size - 1}): ")
        from_y = get_player_input(f"  Y座標 (0-{board_size - 1}): ")
        
        # 有効な移動先を表示（タイル配置後の状態を考慮）
        # 一時的にタイルを配置して移動可能先を確認
        if place_tile and tile_x is not None and tile_y is not None:
            # タイルを一時配置
            old_color = game.board.get_tile_color(tile_x, tile_y)
            game.board.set_tile_color(tile_x, tile_y, tile_color)
            valid_moves = game.get_valid_moves(from_x, from_y)
            # 元に戻す
            game.board.set_tile_color(tile_x, tile_y, old_color)
        else:
            valid_moves = game.get_valid_moves(from_x, from_y)
        
        if not valid_moves:
            print("そのコマは移動できません。別のコマを選択してください。")
            continue
        
        print(f"\n移動可能な位置: {valid_moves}")
        
        # 移動先の選択
        print("移動先を選択してください")
        to_x = get_player_input(f"  X座標 (0-{board_size - 1}): ")
        to_y = get_player_input(f"  Y座標 (0-{board_size - 1}): ")
        
        # 移動を実行
        if game.make_move(from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color):
            print("\n移動成功！")
        else:
            print("\n無効な移動です。もう一度試してください。")
    
    # ゲーム終了
    game.display_game_state()
    print("\n" + "="*60)
    print(f"おめでとうございます！Player {game.winner.value} の勝利です！")
    print("="*60)
    
    # 移動履歴の表示
    print("\n移動履歴:")
    for i, move in enumerate(game.move_history, 1):
        player = move['player']
        from_pos = move['from']
        to_pos = move['to']
        captured = move['captured']
        tile_placed = move['tile_placed']
        
        msg = f"{i}. Player {player.value}: {from_pos} → {to_pos}"
        if tile_placed:
            tile_x, tile_y, tile_col = tile_placed
            msg += f" [タイル配置: {tile_col.value} at ({tile_x},{tile_y})]"
        if captured:
            msg += f" (Player {captured.owner.value} のコマを獲得)"
        print(msg)


if __name__ == "__main__":
    try:
        play_interactive_game()
    except KeyboardInterrupt:
        print("\n\nゲームを終了しました。")
