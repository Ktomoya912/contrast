"""
コントラストゲームのGUIアプリケーション
Tkinterを使用したグラフィカルインターフェース
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Tuple, List
import copy

from contrast_game import ContrastGame, Player, TileColor
from td_learning import TDLearner


class ContrastGUI:
    """コントラストゲームのGUIクラス"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("コントラスト - ボードゲーム")
        self.root.resizable(False, False)
        
        # ゲーム設定
        self.board_size = 5
        self.cell_size = 80
        self.piece_radius = 25
        
        # ゲーム状態
        self.game = ContrastGame(self.board_size)
        self.game.setup_initial_position()
        
        # AI設定
        self.ai_enabled = False
        self.ai_player = None
        self.ai_learner = None
        
        # 選択状態
        self.selected_piece: Optional[Tuple[int, int]] = None
        self.selected_tile_color: Optional[TileColor] = None
        self.tile_to_place: Optional[Tuple[int, int]] = None
        self.valid_moves: List[Tuple[int, int]] = []
        
        # 色設定
        self.colors = {
            'white_tile': '#F5F5F5',
            'black_tile': '#333333',
            'gray_tile': '#888888',
            'grid': '#CCCCCC',
            'player1': '#4A90E2',
            'player2': '#E24A4A',
            'selected': '#FFD700',
            'valid_move': '#90EE90',
            'highlight': '#FFA500'
        }
        
        # UIを構築
        self.create_widgets()
        self.update_display()
    
    def create_widgets(self):
        """UIコンポーネントを作成"""
        # メインフレーム
        main_frame = tk.Frame(self.root, bg='#FFFFFF')
        main_frame.pack(padx=10, pady=10)
        
        # 左側：情報パネル
        left_frame = tk.Frame(main_frame, bg='#FFFFFF')
        left_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # ゲーム情報
        info_frame = tk.LabelFrame(left_frame, text="ゲーム情報", padx=10, pady=10, bg='#FFFFFF')
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.current_player_label = tk.Label(info_frame, text="", font=('Arial', 12, 'bold'), bg='#FFFFFF')
        self.current_player_label.pack(pady=5)
        
        # タイル情報
        tile_frame = tk.LabelFrame(left_frame, text="タイル残数", padx=10, pady=10, bg='#FFFFFF')
        tile_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.p1_black_label = tk.Label(tile_frame, text="", font=('Arial', 10), bg='#FFFFFF')
        self.p1_black_label.pack(anchor=tk.W)
        
        self.p1_gray_label = tk.Label(tile_frame, text="", font=('Arial', 10), bg='#FFFFFF')
        self.p1_gray_label.pack(anchor=tk.W)
        
        tk.Label(tile_frame, text="", bg='#FFFFFF').pack(pady=2)
        
        self.p2_black_label = tk.Label(tile_frame, text="", font=('Arial', 10), bg='#FFFFFF')
        self.p2_black_label.pack(anchor=tk.W)
        
        self.p2_gray_label = tk.Label(tile_frame, text="", font=('Arial', 10), bg='#FFFFFF')
        self.p2_gray_label.pack(anchor=tk.W)
        
        # タイル選択ボタン
        tile_select_frame = tk.LabelFrame(left_frame, text="タイル配置", padx=10, pady=10, bg='#FFFFFF')
        tile_select_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.black_tile_btn = tk.Button(
            tile_select_frame, 
            text="黒タイル (■)",
            command=lambda: self.select_tile(TileColor.BLACK),
            width=15,
            height=2
        )
        self.black_tile_btn.pack(pady=5)
        
        self.gray_tile_btn = tk.Button(
            tile_select_frame,
            text="グレータイル (▦)",
            command=lambda: self.select_tile(TileColor.GRAY),
            width=15,
            height=2
        )
        self.gray_tile_btn.pack(pady=5)
        
        self.cancel_tile_btn = tk.Button(
            tile_select_frame,
            text="配置キャンセル",
            command=self.cancel_tile_placement,
            width=15,
            state=tk.DISABLED
        )
        self.cancel_tile_btn.pack(pady=5)
        
        # 操作説明
        help_frame = tk.LabelFrame(left_frame, text="操作方法", padx=10, pady=10, bg='#FFFFFF')
        help_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        help_text = """
1. タイルを配置する場合は
   ボタンを押してからマスをクリック

2. 移動するコマをクリック

3. 移動先をクリック

4. 緑色のマスが移動可能な場所
        """
        tk.Label(help_frame, text=help_text, justify=tk.LEFT, bg='#FFFFFF', font=('Arial', 9)).pack()
        
        # コントロールボタン
        control_frame = tk.Frame(left_frame, bg='#FFFFFF')
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        self.reset_btn = tk.Button(
            control_frame,
            text="ゲームリセット",
            command=self.reset_game,
            width=15,
            height=2
        )
        self.reset_btn.pack(pady=5)
        
        self.ai_toggle_btn = tk.Button(
            control_frame,
            text="AI有効化",
            command=self.toggle_ai,
            width=15,
            height=2
        )
        self.ai_toggle_btn.pack(pady=5)
        
        # 右側：ボード
        board_frame = tk.Frame(main_frame, bg='#FFFFFF')
        board_frame.pack(side=tk.LEFT)
        
        # キャンバス作成
        canvas_size = self.cell_size * self.board_size
        self.canvas = tk.Canvas(
            board_frame,
            width=canvas_size,
            height=canvas_size,
            bg='#FFFFFF',
            highlightthickness=2,
            highlightbackground='#333333'
        )
        self.canvas.pack()
        
        # クリックイベント
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        # ステータスバー
        self.status_label = tk.Label(
            self.root,
            text="Player 1 のターン",
            font=('Arial', 10),
            bg='#F0F0F0',
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def draw_board(self):
        """ボードを描画"""
        self.canvas.delete('all')
        
        # タイルを描画
        for y in range(self.board_size):
            for x in range(self.board_size):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # タイルの色
                tile_color = self.game.board.get_tile_color(x, y)
                if tile_color == TileColor.WHITE:
                    fill = self.colors['white_tile']
                elif tile_color == TileColor.BLACK:
                    fill = self.colors['black_tile']
                else:
                    fill = self.colors['gray_tile']
                
                # タイル配置予定地をハイライト
                if self.tile_to_place == (x, y):
                    fill = self.colors['highlight']
                
                # 有効な移動先をハイライト
                if (x, y) in self.valid_moves:
                    fill = self.colors['valid_move']
                
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill,
                    outline=self.colors['grid'],
                    width=2
                )
                
                # 座標を表示
                if y == 0:
                    self.canvas.create_text(
                        x1 + self.cell_size // 2, 10,
                        text=str(x),
                        font=('Arial', 8),
                        fill='#666666'
                    )
                if x == 0:
                    self.canvas.create_text(
                        10, y1 + self.cell_size // 2,
                        text=str(y),
                        font=('Arial', 8),
                        fill='#666666'
                    )
        
        # コマを描画
        for y in range(self.board_size):
            for x in range(self.board_size):
                piece = self.game.board.get_piece(x, y)
                if piece:
                    cx = x * self.cell_size + self.cell_size // 2
                    cy = y * self.cell_size + self.cell_size // 2
                    
                    # コマの色
                    if piece.owner == Player.PLAYER1:
                        color = self.colors['player1']
                    else:
                        color = self.colors['player2']
                    
                    # 選択されたコマをハイライト
                    outline_color = color
                    outline_width = 2
                    if self.selected_piece == (x, y):
                        outline_color = self.colors['selected']
                        outline_width = 4
                    
                    # コマを描画
                    self.canvas.create_oval(
                        cx - self.piece_radius,
                        cy - self.piece_radius,
                        cx + self.piece_radius,
                        cy + self.piece_radius,
                        fill=color,
                        outline=outline_color,
                        width=outline_width
                    )
                    
                    # プレイヤー番号を表示
                    self.canvas.create_text(
                        cx, cy,
                        text=str(piece.owner.value),
                        font=('Arial', 16, 'bold'),
                        fill='white'
                    )
    
    def on_canvas_click(self, event):
        """キャンバスクリック時の処理"""
        if self.game.game_over:
            return
        
        # AIのターンの場合は操作不可
        if self.ai_enabled and self.game.current_player == self.ai_player:
            return
        
        # クリック位置を座標に変換
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return
        
        # タイル配置モード
        if self.selected_tile_color is not None:
            self.place_tile_at(x, y)
            return
        
        # コマが選択されていない場合
        if self.selected_piece is None:
            piece = self.game.board.get_piece(x, y)
            if piece and piece.owner == self.game.current_player:
                self.select_piece(x, y)
        else:
            # 移動先を選択
            if (x, y) in self.valid_moves:
                self.move_piece_to(x, y)
            else:
                # 別のコマを選択
                piece = self.game.board.get_piece(x, y)
                if piece and piece.owner == self.game.current_player:
                    self.select_piece(x, y)
                else:
                    self.deselect_piece()
    
    def select_piece(self, x: int, y: int):
        """コマを選択"""
        self.selected_piece = (x, y)
        
        # タイル配置を考慮した有効な移動先を取得
        if self.tile_to_place:
            temp_game = copy.deepcopy(self.game)
            tx, ty = self.tile_to_place
            temp_game.board.set_tile_color(tx, ty, self.selected_tile_color)
            self.valid_moves = temp_game.get_valid_moves(x, y)
        else:
            self.valid_moves = self.game.get_valid_moves(x, y)
        
        self.update_display()
        self.status_label.config(text=f"移動先を選択してください（移動可能: {len(self.valid_moves)}箇所）")
    
    def deselect_piece(self):
        """コマの選択を解除"""
        self.selected_piece = None
        self.valid_moves = []
        self.update_display()
    
    def select_tile(self, tile_color: TileColor):
        """タイルを選択"""
        tiles_remaining = self.game.tiles_remaining[self.game.current_player]
        
        if tile_color == TileColor.BLACK and tiles_remaining['black'] <= 0:
            messagebox.showwarning("警告", "黒タイルが残っていません")
            return
        
        if tile_color == TileColor.GRAY and tiles_remaining['gray'] <= 0:
            messagebox.showwarning("警告", "グレータイルが残っていません")
            return
        
        self.selected_tile_color = tile_color
        self.cancel_tile_btn.config(state=tk.NORMAL)
        
        tile_name = "黒タイル" if tile_color == TileColor.BLACK else "グレータイル"
        self.status_label.config(text=f"{tile_name}を配置する位置を選択してください")
        
        self.update_tile_buttons()
    
    def place_tile_at(self, x: int, y: int):
        """タイルを配置"""
        if self.game.board.get_tile_color(x, y) != TileColor.WHITE:
            messagebox.showwarning("警告", "そこにはタイルを配置できません")
            return
        
        self.tile_to_place = (x, y)
        self.selected_tile_color = None
        self.cancel_tile_btn.config(state=tk.NORMAL)
        
        self.status_label.config(text="移動するコマを選択してください")
        self.update_tile_buttons()
        self.update_display()
    
    def cancel_tile_placement(self):
        """タイル配置をキャンセル"""
        self.selected_tile_color = None
        self.tile_to_place = None
        self.cancel_tile_btn.config(state=tk.DISABLED)
        self.status_label.config(text=f"Player {self.game.current_player.value} のターン")
        self.update_tile_buttons()
        self.update_display()
    
    def move_piece_to(self, to_x: int, to_y: int):
        """コマを移動"""
        from_x, from_y = self.selected_piece
        
        # タイル配置情報
        place_tile = self.tile_to_place is not None
        tile_x = self.tile_to_place[0] if place_tile else None
        tile_y = self.tile_to_place[1] if place_tile else None
        tile_color = self.selected_tile_color if place_tile else None
        
        # 移動を実行
        success = self.game.make_move(
            from_x, from_y, to_x, to_y,
            place_tile, tile_x, tile_y, tile_color
        )
        
        if success:
            self.selected_piece = None
            self.valid_moves = []
            self.tile_to_place = None
            self.selected_tile_color = None
            self.cancel_tile_btn.config(state=tk.DISABLED)
            
            self.update_display()
            
            # ゲーム終了チェック
            if self.game.game_over:
                self.show_game_over()
            elif self.ai_enabled and self.game.current_player == self.ai_player:
                # AIのターン
                self.root.after(500, self.ai_move)
        else:
            messagebox.showerror("エラー", "移動に失敗しました")
    
    def ai_move(self):
        """AIの手を実行"""
        if not self.ai_enabled or self.game.game_over:
            return
        
        self.status_label.config(text="AIが考えています...")
        self.root.update()
        
        try:
            action = self.ai_learner.select_action(self.game, use_epsilon=False)
            
            if action:
                from_x, from_y, to_x, to_y, place_tile, tile_x, tile_y, tile_color = action
                
                self.game.make_move(
                    from_x, from_y, to_x, to_y,
                    place_tile, tile_x, tile_y, tile_color
                )
                
                self.update_display()
                
                if self.game.game_over:
                    self.show_game_over()
            else:
                messagebox.showinfo("情報", "AIに有効な手がありません")
        except Exception as e:
            messagebox.showerror("エラー", f"AI実行エラー: {str(e)}")
    
    def update_display(self):
        """表示を更新"""
        self.draw_board()
        
        # プレイヤー情報を更新
        current = self.game.current_player.value
        self.current_player_label.config(
            text=f"Player {current} のターン",
            fg=self.colors[f'player{current}']
        )
        
        # タイル情報を更新
        p1_tiles = self.game.tiles_remaining[Player.PLAYER1]
        p2_tiles = self.game.tiles_remaining[Player.PLAYER2]
        
        self.p1_black_label.config(text=f"Player 1 黒タイル: {p1_tiles['black']}")
        self.p1_gray_label.config(text=f"Player 1 グレータイル: {p1_tiles['gray']}")
        self.p2_black_label.config(text=f"Player 2 黒タイル: {p2_tiles['black']}")
        self.p2_gray_label.config(text=f"Player 2 グレータイル: {p2_tiles['gray']}")
        
        # タイルボタンの状態を更新
        self.update_tile_buttons()
        
        # ステータスバーを更新
        if not self.game.game_over:
            if self.tile_to_place:
                self.status_label.config(text="移動するコマを選択してください")
            elif self.selected_piece:
                self.status_label.config(text="移動先を選択してください")
            else:
                self.status_label.config(text=f"Player {current} のターン")
    
    def update_tile_buttons(self):
        """タイルボタンの状態を更新"""
        tiles = self.game.tiles_remaining[self.game.current_player]
        
        # AIのターンの場合は無効化
        if self.ai_enabled and self.game.current_player == self.ai_player:
            self.black_tile_btn.config(state=tk.DISABLED)
            self.gray_tile_btn.config(state=tk.DISABLED)
            return
        
        # タイル配置中または既に配置済みの場合は無効化
        if self.tile_to_place or self.selected_tile_color:
            self.black_tile_btn.config(state=tk.DISABLED)
            self.gray_tile_btn.config(state=tk.DISABLED)
        else:
            self.black_tile_btn.config(
                state=tk.NORMAL if tiles['black'] > 0 else tk.DISABLED
            )
            self.gray_tile_btn.config(
                state=tk.NORMAL if tiles['gray'] > 0 else tk.DISABLED
            )
    
    def show_game_over(self):
        """ゲーム終了メッセージを表示"""
        winner = self.game.winner
        if winner:
            message = f"Player {winner.value} の勝利！"
            self.status_label.config(text=message)
            messagebox.showinfo("ゲーム終了", message)
        else:
            message = "引き分け"
            self.status_label.config(text=message)
            messagebox.showinfo("ゲーム終了", message)
    
    def reset_game(self):
        """ゲームをリセット"""
        if messagebox.askyesno("確認", "ゲームをリセットしますか？"):
            self.game = ContrastGame(self.board_size)
            self.game.setup_initial_position()
            
            self.selected_piece = None
            self.selected_tile_color = None
            self.tile_to_place = None
            self.valid_moves = []
            
            self.cancel_tile_btn.config(state=tk.DISABLED)
            
            self.update_display()
    
    def toggle_ai(self):
        """AIを有効/無効化"""
        if not self.ai_enabled:
            # AI有効化
            self.load_ai()
        else:
            # AI無効化
            self.ai_enabled = False
            self.ai_player = None
            self.ai_learner = None
            self.ai_toggle_btn.config(text="AI有効化")
            self.status_label.config(text="AIを無効化しました")
    
    def load_ai(self):
        """AIを読み込み"""
        # プレイヤー選択ダイアログ
        dialog = tk.Toplevel(self.root)
        dialog.title("AI設定")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="AIがプレイするプレイヤーを選択", font=('Arial', 11)).pack(pady=20)
        
        player_var = tk.IntVar(value=2)
        
        tk.Radiobutton(dialog, text="Player 1", variable=player_var, value=1).pack()
        tk.Radiobutton(dialog, text="Player 2", variable=player_var, value=2).pack()
        
        def confirm():
            player_num = player_var.get()
            dialog.destroy()
            
            try:
                self.ai_learner = TDLearner(board_size=5, epsilon=0.0, use_cuda=False)
                try:
                    self.ai_learner.value_network.load("contrast_ai.pth")
                    status_msg = "学習済みAIを読み込みました"
                except:
                    status_msg = "未学習のAIを使用します"
                
                self.ai_enabled = True
                self.ai_player = Player.PLAYER1 if player_num == 1 else Player.PLAYER2
                self.ai_toggle_btn.config(text="AI無効化")
                self.status_label.config(text=status_msg)
                
                # AIが先手の場合、すぐに手を打つ
                if self.game.current_player == self.ai_player:
                    self.root.after(500, self.ai_move)
                
            except Exception as e:
                messagebox.showerror("エラー", f"AI読み込みエラー: {str(e)}")
        
        tk.Button(dialog, text="OK", command=confirm, width=10).pack(pady=10)


def main():
    """メイン関数"""
    root = tk.Tk()
    app = ContrastGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
