"""
AIと対戦できるインタラクティブゲーム
"""

from contrast_game import ContrastGame, Player, TileColor
from td_learning import TDLearner


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
        if response in ["y", "yes", "はい", "h"]:
            return True
        elif response in ["n", "no", "いいえ", "i"]:
            return False
        else:
            print("y/n で答えてください。")


def ai_move(game: ContrastGame, learner: TDLearner) -> bool:
    """
    AIの手を実行

    Returns:
        成功したかどうか
    """
    print("\nAIが考慮中...")

    # AIの行動を選択（ε-greedyなし）
    action = learner.select_action(game, use_epsilon=False)

    if action is None:
        print("AIに有効な手がありません")
        return False

    from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color = action

    # タイル配置の表示
    if place_tile:
        tile_name = "黒タイル" if tile_color == TileColor.BLACK else "グレータイル"
        print(f"AI: {tile_name}を({tile_x}, {tile_y})に配置")

    # 移動の表示
    print(f"AI: コマを({from_x}, {from_y})から({to_x}, {to_y})へ移動")

    # 手を実行
    success = game.make_move(
        from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
    )

    return success


def play_vs_ai(model_path: str = "contrast_ai.pth", use_cuda: bool = False):
    """AIと対戦"""
    print("=" * 60)
    print("コントラスト - AI対戦モード")
    print("=" * 60)
    print()

    # AIを読み込み
    print("AIを読み込んでいます...")
    learner = TDLearner(
        board_size=5, epsilon=0.0, use_cuda=use_cuda
    )  # ε=0（常に最良手を選択）

    try:
        learner.value_network.load(model_path)
    except FileNotFoundError:
        print(f"警告: モデルファイル '{model_path}' が見つかりません")
        print("未学習のAIと対戦します")

    print("AIの準備完了！")
    print()

    # プレイヤーの色を選択
    print("あなたはどちらのプレイヤーですか？")
    print("1: プレイヤー1（下側スタート、先手）")
    print("2: プレイヤー2（上側スタート、後手）")

    while True:
        choice = input("選択 (1/2): ").strip()
        if choice == "1":
            human_player = Player.PLAYER1
            ai_player = Player.PLAYER2
            break
        elif choice == "2":
            human_player = Player.PLAYER2
            ai_player = Player.PLAYER1
            break
        else:
            print("1 または 2 を入力してください")

    print(f"\nあなた: Player {human_player.value}")
    print(f"AI: Player {ai_player.value}")
    print()

    # ゲーム開始
    board_size = 5
    game = ContrastGame(board_size=board_size)
    game.setup_initial_position()

    # ゲームループ
    while not game.game_over:
        # 現在の状態を表示
        game.display_game_state()

        if game.current_player == human_player:
            # 人間のターン
            print(f"\nあなたのターン (Player {human_player.value})")

            # タイルを配置するか確認
            place_tile = False
            tile_x = None
            tile_y = None
            tile_color = None

            tiles = game.tiles_remaining[game.current_player]
            if tiles["black"] > 0 or tiles["gray"] > 0:
                if get_yes_no_input("タイルを配置しますか？ (y/n): "):
                    place_tile = True

                    # タイルの種類を選択
                    while True:
                        print(
                            f"残りタイル - 黒(■): {tiles['black']}, グレー(▦): {tiles['gray']}"
                        )
                        tile_type = (
                            input("配置するタイルの種類 (black/gray): ").strip().lower()
                        )

                        if tile_type in ["black", "b", "黒"] and tiles["black"] > 0:
                            tile_color = TileColor.BLACK
                            break
                        elif (
                            tile_type in ["gray", "grey", "g", "グレー"]
                            and tiles["gray"] > 0
                        ):
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

            # 有効な移動先を表示
            if place_tile and tile_x is not None and tile_y is not None:
                old_color = game.board.get_tile_color(tile_x, tile_y)
                game.board.set_tile_color(tile_x, tile_y, tile_color)
                valid_moves = game.get_valid_moves(from_x, from_y)
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
            if game.make_move(
                from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
            ):
                print("\n移動成功！")
            else:
                print("\n無効な移動です。もう一度試してください。")

        else:
            # AIのターン
            print(f"\nAIのターン (Player {ai_player.value})")

            if not ai_move(game, learner):
                print("AIの手の実行に失敗しました")
                break

            input("\nEnterキーを押して続行...")

    # ゲーム終了
    game.display_game_state()
    print("\n" + "=" * 60)

    if game.winner == human_player:
        print("おめでとうございます！あなたの勝ちです！")
    elif game.winner == ai_player:
        print("AIの勝利です。次は頑張りましょう！")
    else:
        print("引き分けです。")

    print("=" * 60)


def watch_ai_vs_ai(
    model_path: str = "contrast_ai.pth", num_games: int = 5, use_cuda: bool = False
):
    """AI同士の対戦を観戦"""
    print("=" * 60)
    print("AI同士の対戦を観戦")
    print("=" * 60)
    print()

    # AIを読み込み
    learner = TDLearner(board_size=5, epsilon=0.0, use_cuda=use_cuda)

    try:
        learner.value_network.load(model_path)
        print("学習済みモデルを読み込みました")
    except FileNotFoundError:
        print(f"警告: モデルファイル '{model_path}' が見つかりません")
        print("未学習のAIで対戦します")

    print()

    player1_wins = 0
    player2_wins = 0

    for game_num in range(num_games):
        print(f"\n{'=' * 60}")
        print(f"ゲーム {game_num + 1}/{num_games}")
        print("=" * 60)

        game = ContrastGame(board_size=5)
        game.setup_initial_position()

        move_count = 0
        max_moves = 200

        while not game.game_over and move_count < max_moves:
            game.display_game_state()

            # AIの手を実行
            if not ai_move(game, learner):
                break

            move_count += 1

            if not game.game_over:
                input("\nEnterキーで次の手へ...")

        # 結果を表示
        game.display_game_state()

        if game.winner == Player.PLAYER1:
            print("\nPlayer 1 の勝利！")
            player1_wins += 1
        elif game.winner == Player.PLAYER2:
            print("\nPlayer 2 の勝利！")
            player2_wins += 1
        else:
            print("\n引き分け")

        print(f"手数: {move_count}")

    # 総合結果
    print("\n" + "=" * 60)
    print("総合結果")
    print("=" * 60)
    print(f"Player 1: {player1_wins}勝")
    print(f"Player 2: {player2_wins}勝")
    print(f"引き分け: {num_games - player1_wins - player2_wins}")


if __name__ == "__main__":
    print("コントラスト AI対戦プログラム")
    print("=" * 60)
    print("1: AIと対戦")
    print("2: AI同士の対戦を観戦")
    print("=" * 60)

    choice = input("\n選択 (1/2): ").strip()

    # CUDA使用の確認
    import torch

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda_input = input("CUDAを使用しますか？ (y/n): ").strip().lower()
        use_cuda = use_cuda_input in ["y", "yes"]

    if choice == "1":
        play_vs_ai("contrast_ai.pth", use_cuda=use_cuda)
    elif choice == "2":
        num_games = int(input("何ゲーム観戦しますか？: "))
        watch_ai_vs_ai("contrast_ai.pth", num_games, use_cuda=use_cuda)
    else:
        print("無効な選択です")
