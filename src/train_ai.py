"""
コントラストAIの学習実行スクリプト
"""

import argparse

import matplotlib.pyplot as plt

from td_learning import TDLearner


def plot_training_progress(
    learner: TDLearner, save_path: str = "training_progress.png"
):
    """学習の進捗をグラフ化"""
    if not learner.value_network.loss_history:
        print("学習履歴がありません")
        return

    plt.figure(figsize=(12, 5))

    # 損失の推移
    plt.subplot(1, 2, 1)
    plt.plot(learner.value_network.loss_history)
    plt.title("Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)

    # 勝率の推移（移動平均）
    plt.subplot(1, 2, 2)

    if learner.game_count > 0:
        win_rate_p1 = learner.player1_wins / learner.game_count
        win_rate_p2 = learner.player2_wins / learner.game_count

        plt.bar(["Player 1", "Player 2"], [win_rate_p1, win_rate_p2])
        plt.title("Win Rates")
        plt.ylabel("Win Rate")
        plt.ylim(0, 1)
        plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"学習グラフを保存しました: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="コントラストAIの学習")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="学習エピソード数 (デフォルト: 1000)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="学習率 (デフォルト: 0.001)"
    )
    parser.add_argument(
        "--discount", type=float, default=0.95, help="割引率 (デフォルト: 0.95)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="ε-greedyのε (デフォルト: 0.2)"
    )
    parser.add_argument(
        "--lambda-td", type=float, default=0.7, help="TD(λ)のλ (デフォルト: 0.7)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="モデル保存間隔 (デフォルト: 100)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="contrast_ai.pth",
        help="モデルの保存パス (デフォルト: contrast_ai.pth)",
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="既存モデルを読み込んで学習を継続"
    )
    parser.add_argument("--plot", action="store_true", help="学習進捗をグラフ化")
    parser.add_argument(
        "--cuda", action="store_true", help="CUDAを使用する（GPUがある場合）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("コントラストAI - TD学習")
    print("=" * 60)
    print(f"エピソード数: {args.episodes}")
    print(f"学習率: {args.learning_rate}")
    print(f"割引率: {args.discount}")
    print(f"ε: {args.epsilon}")
    print(f"λ: {args.lambda_td}")
    print(f"保存パス: {args.model_path}")
    print(f"CUDA使用: {args.cuda}")
    print("=" * 60)
    print()

    # 学習器を初期化
    learner = TDLearner(
        board_size=5,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        epsilon=args.epsilon,
        lambda_td=args.lambda_td,
        use_cuda=args.cuda,
    )

    # 既存モデルを読み込む場合
    if args.load_model:
        try:
            learner.value_network.load(args.load_model)
            print(f"モデルを読み込みました: {args.load_model}")
            print()
        except FileNotFoundError:
            print(f"警告: モデルファイル '{args.load_model}' が見つかりません")
            print("新規に学習を開始します")
            print()

    # 学習を実行
    try:
        learner.train(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
        )
    except KeyboardInterrupt:
        print("学習を中断しました")
        save = input("現在のモデルを保存しますか？ (y/n): ").strip().lower()
        if save in ["y", "yes"]:
            learner.value_network.save(args.model_path)
            print("モデルを保存しました")

    # 学習進捗をグラフ化
    if args.plot:
        plot_training_progress(learner, "training_progress.png")

    print("学習完了！")
    print(f"モデルは {args.model_path} に保存されました")
    print("次のコマンドでAIと対戦できます:")
    print("  python3 play_with_ai.py")


if __name__ == "__main__":
    main()
