"""
コントラストゲームのWeb GUI
FlaskベースのWebアプリケーション（WSL対応）
"""

import logging
import os

from flask import Flask, jsonify, render_template, request

from ac_learning import ActorCriticLearner
from contrast_game import ContrastGame, Player, TileColor
from rule_based_player import RuleBasedPlayer

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "contrast_game_secret_key"

# ゲームセッション（簡易版：本来はセッション管理が必要）
game_state = {
    "game": None,
    "ai_enabled": False,
    "ai_player": None,
    "ai_learner": None,
    "rule_enabled": False,
    "rule_player": None,
    "rule_player_obj": None,
    "selected_piece": None,
    "selected_tile_color": None,
    "tile_to_place": None,
    "move_made": False,  # コマ移動完了フラグ
    "moved_from": None,  # 移動元の座標
    "moved_to": None,  # 移動先の座標
}


def init_game():
    """ゲームを初期化"""
    game_state["game"] = ContrastGame(board_size=5)
    game_state["game"].setup_initial_position()
    game_state["selected_piece"] = None
    game_state["selected_tile_color"] = None
    game_state["tile_to_place"] = None
    game_state["move_made"] = False
    game_state["moved_from"] = None
    game_state["moved_to"] = None


def get_game_data():
    """ゲーム状態をJSON形式で取得（NumPy int64対応）"""
    game = game_state["game"]

    # ボード状態
    board = []
    for y in range(game.board.size):
        row = []
        for x in range(game.board.size):
            piece = game.board.get_piece(x, y)
            tile = game.board.get_tile_color(x, y)

            cell = {
                "x": int(x),  # int64からPython intに変換
                "y": int(y),
                "tile": tile.value,  # enum値（文字列）をそのまま返す
                "piece": int(piece.owner.value) if piece else None,
            }
            row.append(cell)
        board.append(row)

    return {
        "board": board,
        "current_player": int(game.current_player.value),
        "game_over": game.game_over,
        "winner": int(game.winner.value) if game.winner else None,
        "tiles_remaining": {
            "player1": game.tiles_remaining[Player.PLAYER1],
            "player2": game.tiles_remaining[Player.PLAYER2],
        },
        "selected_piece": game_state["selected_piece"],
        "selected_tile_color": game_state["selected_tile_color"].value
        if game_state["selected_tile_color"]
        else None,
        "tile_to_place": game_state["tile_to_place"],
        "ai_enabled": game_state["ai_enabled"],
        "ai_player": int(game_state["ai_player"].value)
        if game_state["ai_player"]
        else None,
        "rule_enabled": game_state["rule_enabled"],
        "rule_player": int(game_state["rule_player"].value)
        if game_state["rule_player"]
        else None,
        "move_made": game_state["move_made"],
        "moved_from": game_state["moved_from"],
        "moved_to": game_state["moved_to"],
    }


@app.route("/")
def index():
    """メインページ"""
    return render_template("index.html")


@app.route("/api/init", methods=["POST"])
def api_init():
    """ゲームを初期化"""
    init_game()
    return jsonify({"success": True, "data": get_game_data()})


@app.route("/api/state", methods=["GET"])
def api_state():
    """現在のゲーム状態を取得"""
    if game_state["game"] is None:
        init_game()
    return jsonify({"success": True, "data": get_game_data()})


@app.route("/api/valid_moves", methods=["POST"])
def api_valid_moves():
    """指定位置のコマの有効な移動先を取得"""
    data = request.json
    x = data["x"]
    y = data["y"]

    game = game_state["game"]
    valid_moves = game.get_valid_moves(x, y)
    # NumPy int64をPython intに変換
    valid_moves = [(int(mx), int(my)) for mx, my in valid_moves]

    return jsonify({"success": True, "valid_moves": valid_moves})


@app.route("/api/select_piece", methods=["POST"])
def api_select_piece():
    """コマを選択"""
    data = request.json
    x = data["x"]
    y = data["y"]

    game = game_state["game"]
    piece = game.board.get_piece(x, y)

    if piece and piece.owner == game.current_player:
        game_state["selected_piece"] = (x, y)
        valid_moves = game.get_valid_moves(x, y)
        # NumPy int64をPython intに変換
        valid_moves = [(int(mx), int(my)) for mx, my in valid_moves]
        return jsonify(
            {"success": True, "valid_moves": valid_moves, "data": get_game_data()}
        )
    else:
        return jsonify({"success": False, "error": "そのコマは選択できません"})


@app.route("/api/move", methods=["POST"])
def api_move():
    """コマを移動（ターンは終了せず、タイル配置の選択肢を提供）"""
    data = request.json
    to_x = data["to_x"]
    to_y = data["to_y"]

    if game_state["selected_piece"] is None:
        return jsonify({"success": False, "error": "コマが選択されていません"})

    from_x, from_y = game_state["selected_piece"]
    game = game_state["game"]

    # 移動が有効かチェック
    valid_moves = game.get_valid_moves(from_x, from_y)
    if (to_x, to_y) not in valid_moves:
        return jsonify({"success": False, "error": "無効な移動です"})

    # 移動を一時的に記録（まだゲームには反映しない）
    game_state["move_made"] = True
    game_state["moved_from"] = (from_x, from_y)
    game_state["moved_to"] = (to_x, to_y)
    game_state["selected_piece"] = None

    return jsonify(
        {
            "success": True,
            "data": get_game_data(),
            "message": "コマを移動しました。ターン終了またはタイル配置を選択してください",
        }
    )


@app.route("/api/end_turn", methods=["POST"])
def api_end_turn():
    """ターン終了（タイル配置なしまたはタイル配置後）"""
    if not game_state["move_made"]:
        return jsonify({"success": False, "error": "コマを移動してください"})

    game = game_state["game"]
    from_x, from_y = game_state["moved_from"]
    to_x, to_y = game_state["moved_to"]

    # タイル配置情報
    place_tile = (
        game_state["tile_to_place"] is not None
        and game_state["selected_tile_color"] is not None
    )
    tile_x = game_state["tile_to_place"][0] if place_tile else None
    tile_y = game_state["tile_to_place"][1] if place_tile else None
    tile_color = game_state["selected_tile_color"] if place_tile else None

    # 実際にゲームに移動を反映
    success = game.make_move(
        from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
    )

    if success:
        # 状態をリセット
        game_state["move_made"] = False
        game_state["moved_from"] = None
        game_state["moved_to"] = None
        game_state["selected_piece"] = None
        game_state["selected_tile_color"] = None
        game_state["tile_to_place"] = None

        # AI or ルールベースのターンかチェック
        ai_move_data = None
        rule_move_data = None
        if not game.game_over:
            if (
                game_state["ai_enabled"]
                and game.current_player == game_state["ai_player"]
            ):
                ai_move_data = execute_ai_move()
            elif (
                game_state["rule_enabled"]
                and game.current_player == game_state["rule_player"]
            ):
                rule_move_data = execute_rule_move()

        return jsonify(
            {
                "success": True,
                "data": get_game_data(),
                "ai_move": ai_move_data,
                "rule_move": rule_move_data,
            }
        )
    else:
        return jsonify({"success": False, "error": "ターン終了に失敗しました"})


@app.route("/api/select_tile", methods=["POST"])
def api_select_tile():
    """タイルを選択"""
    data = request.json
    tile_type = data["tile_type"]

    game = game_state["game"]
    tiles_remaining = game.tiles_remaining[game.current_player]

    if tile_type == "black":
        if tiles_remaining["black"] <= 0:
            return jsonify({"success": False, "error": "黒タイルが残っていません"})
        game_state["selected_tile_color"] = TileColor.BLACK
    elif tile_type == "gray":
        if tiles_remaining["gray"] <= 0:
            return jsonify({"success": False, "error": "グレータイルが残っていません"})
        game_state["selected_tile_color"] = TileColor.GRAY
    else:
        return jsonify({"success": False, "error": "無効なタイルタイプ"})

    return jsonify({"success": True, "data": get_game_data()})


@app.route("/api/place_tile", methods=["POST"])
def api_place_tile():
    """タイル配置位置を記録して自動的にターン終了"""
    if not game_state["move_made"]:
        return jsonify({"success": False, "error": "まずコマを移動してください"})

    if game_state["selected_tile_color"] is None:
        return jsonify({"success": False, "error": "タイルを選択してください"})

    data = request.json
    x = data["x"]
    y = data["y"]

    game = game_state["game"]

    # コマやタイルがある場所には配置できない
    if game.board.get_piece(x, y) is not None:
        return jsonify(
            {"success": False, "error": "コマがある場所にはタイルを配置できません"}
        )

    if game.board.get_tile_color(x, y) != TileColor.WHITE:
        return jsonify({"success": False, "error": "既にタイルが配置されています"})

    # タイル配置位置を記録
    game_state["tile_to_place"] = (x, y)

    # 自動的にターン終了
    from_x, from_y = game_state["moved_from"]
    to_x, to_y = game_state["moved_to"]
    tile_color = game_state["selected_tile_color"]

    # 実際にゲームに移動を反映
    success = game.make_move(from_x, from_y, to_x, to_y, True, x, y, tile_color)

    if success:
        # 状態をリセット
        game_state["move_made"] = False
        game_state["moved_from"] = None
        game_state["moved_to"] = None
        game_state["selected_piece"] = None
        game_state["selected_tile_color"] = None
        game_state["tile_to_place"] = None

        # AI or ルールベースのターンかチェック
        ai_move_data = None
        rule_move_data = None
        if not game.game_over:
            if (
                game_state["ai_enabled"]
                and game.current_player == game_state["ai_player"]
            ):
                ai_move_data = execute_ai_move()
            elif (
                game_state["rule_enabled"]
                and game.current_player == game_state["rule_player"]
            ):
                rule_move_data = execute_rule_move()

        return jsonify(
            {
                "success": True,
                "data": get_game_data(),
                "ai_move": ai_move_data,
                "rule_move": rule_move_data,
                "message": "タイルを配置してターン終了しました",
            }
        )
    else:
        return jsonify({"success": False, "error": "タイル配置に失敗しました"})


@app.route("/api/cancel_tile", methods=["POST"])
def api_cancel_tile():
    """タイル選択をキャンセル"""
    game_state["selected_tile_color"] = None
    return jsonify({"success": True, "data": get_game_data()})


@app.route("/api/toggle_ai", methods=["POST"])
def api_toggle_ai():
    """AIを有効/無効化"""
    data = request.json
    enable = data.get("enable", False)
    player = data.get("player", 2)
    model_path = data.get("model_path", "contrast_ac.pth")

    if enable:
        try:
            # Actor-Critic学習器を初期化
            learner = ActorCriticLearner(
                board_size=5,
                max_actions=500,
                learning_rate=0.0003,
                discount_factor=0.99,
                use_cuda=False,
            )

            # モデルを読み込み
            try:
                learner.network.load(model_path)
                message = f"学習済みAIを読み込みました: {model_path}"
            except FileNotFoundError:
                message = f"警告: {model_path} が見つかりません。未学習のAIを使用します"
            except Exception as e:
                message = "警告: モデル読み込みエラー。未学習のAIを使用します"
                logger.error(f"Model load error: {e}")

            game_state["ai_enabled"] = True
            game_state["ai_player"] = Player.PLAYER1 if player == 1 else Player.PLAYER2
            game_state["ai_learner"] = learner

            # AIが先手の場合、すぐに手を打つ
            ai_move_data = None
            if game_state["game"].current_player == game_state["ai_player"]:
                ai_move_data = execute_ai_move()

            return jsonify(
                {
                    "success": True,
                    "message": message,
                    "data": get_game_data(),
                    "ai_move": ai_move_data,
                }
            )
        except Exception as e:
            return jsonify({"success": False, "error": f"AI読み込みエラー: {str(e)}"})
    else:
        game_state["ai_enabled"] = False
        game_state["ai_player"] = None
        game_state["ai_learner"] = None
        return jsonify(
            {"success": True, "message": "AIを無効化しました", "data": get_game_data()}
        )


@app.route("/api/toggle_rule", methods=["POST"])
def api_toggle_rule():
    """ルールベースプレイヤーを有効/無効化"""
    data = request.json
    enable = data.get("enable", False)
    player = data.get("player", 2)

    if enable:
        try:
            rule_player = Player.PLAYER1 if player == 1 else Player.PLAYER2
            rule_player_obj = RuleBasedPlayer(rule_player)

            game_state["rule_enabled"] = True
            game_state["rule_player"] = rule_player
            game_state["rule_player_obj"] = rule_player_obj

            # ルールベースが先手の場合、すぐに手を打つ
            rule_move_data = None
            if game_state["game"].current_player == game_state["rule_player"]:
                rule_move_data = execute_rule_move()

            return jsonify(
                {
                    "success": True,
                    "message": f"ルールベースプレイヤー (Player {player}) を有効化しました",
                    "data": get_game_data(),
                    "rule_move": rule_move_data,
                }
            )
        except Exception as e:
            return jsonify(
                {"success": False, "error": f"ルールベース読み込みエラー: {str(e)}"}
            )
    else:
        game_state["rule_enabled"] = False
        game_state["rule_player"] = None
        game_state["rule_player_obj"] = None
        return jsonify(
            {
                "success": True,
                "message": "ルールベースを無効化しました",
                "data": get_game_data(),
            }
        )


def execute_ai_move():
    """AIの手を実行"""
    try:
        action_result = game_state["ai_learner"].select_action(game_state["game"])

        if action_result:
            # 新しいインターフェース: (move_index, tile_index, move_tuple, tile_tuple)
            move_index, tile_index, move_tuple, tile_tuple = action_result
            from_x, from_y, to_x, to_y = move_tuple

            # タイル配置情報を処理
            if tile_tuple is None:
                place_tile = False
                tile_x, tile_y, tile_color = None, None, None
            else:
                place_tile = True
                tile_color, tile_x, tile_y = tile_tuple

            game_state["game"].make_move(
                from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
            )

            return {
                "from": (int(from_x), int(from_y)),
                "to": (int(to_x), int(to_y)),
                "tile": (int(tile_x), int(tile_y), tile_color.value)
                if place_tile
                else None,
            }
        else:
            logger.warning("AIに有効な手がありません")
            return None
    except Exception as e:
        logger.error(f"AI実行エラー: {e}")
        return None


def execute_rule_move():
    """ルールベースプレイヤーの手を実行"""
    try:
        action = game_state["rule_player_obj"].select_action(game_state["game"])

        if action:
            move_tuple, tile_tuple = action
            from_x, from_y, to_x, to_y = move_tuple

            # タイル配置情報を処理
            if tile_tuple is None:
                place_tile = False
                tile_x, tile_y, tile_color = None, None, None
            else:
                place_tile = True
                tile_color, tile_x, tile_y = tile_tuple

            game_state["game"].make_move(
                from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color
            )

            return {
                "from": (int(from_x), int(from_y)),
                "to": (int(to_x), int(to_y)),
                "tile": (int(tile_x), int(tile_y), tile_color.value)
                if place_tile
                else None,
            }
        else:
            logger.warning("ルールベースに有効な手がありません")
            return None
    except Exception as e:
        logger.error(f"ルールベース実行エラー: {e}")
        return None


if __name__ == "__main__":
    # テンプレートディレクトリを作成
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(template_dir, exist_ok=True)

    # 初期化
    init_game()

    # サーバー起動
    print("=" * 60)
    print("コントラスト Web GUI")
    print("=" * 60)
    print("ブラウザで http://localhost:5000 を開いてください")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
