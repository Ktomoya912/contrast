import argparse
import logging

import matplotlib.pyplot as plt

from ac_learning import ActorCriticLearner

# ロガー設定
logger = logging.getLogger(__name__)


def plot_training_progress(
    learner: ActorCriticLearner, save_path: str = "training_progress.png"
):
    """学習の進捗をグラフ化"""
    if not learner.network.total_loss_history:
        logger.warning("学習履歴がありません")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Policy Loss
    axes[0, 0].plot(learner.network.policy_loss_history)
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].set_xlabel("Training Steps")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    # Value Loss
    axes[0, 1].plot(learner.network.value_loss_history)
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].set_xlabel("Training Steps")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    # Total Loss
    axes[1, 0].plot(learner.network.total_loss_history)
    axes[1, 0].set_title("Total Loss")
    axes[1, 0].set_xlabel("Training Steps")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)

    # 勝率
    if learner.game_count > 0:
        win_rate_p1 = learner.player1_wins / learner.game_count
        win_rate_p2 = learner.player2_wins / learner.game_count

        axes[1, 1].bar(["Player 1", "Player 2"], [win_rate_p1, win_rate_p2])
        axes[1, 1].set_title("Win Rates")
        axes[1, 1].set_ylabel("Win Rate")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"学習グラフを保存しました: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="コントラストAIの学習 (Actor-Critic)")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="学習エピソード数 (デフォルト: 1000)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="学習率 (デフォルト: 0.0003)",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99, help="割引率 (デフォルト: 0.99)"
    )
    parser.add_argument(
        "--max-actions", type=int, default=500, help="最大行動数 (デフォルト: 500)"
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
        default="contrast_ac.pth",
        help="モデルの保存パス (デフォルト: contrast_ac.pth)",
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="既存モデルを読み込んで学習を継続"
    )
    parser.add_argument("--plot", action="store_true", help="学習進捗をグラフ化")
    parser.add_argument(
        "--cuda", action="store_true", help="CUDAを使用する（GPUがある場合）"
    )

    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.model_path}.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("コントラストAI - Actor-Critic学習")
    logger.info("=" * 60)
    logger.info(f"エピソード数: {args.episodes}")
    logger.info(f"学習率: {args.learning_rate}")
    logger.info(f"割引率: {args.discount}")
    logger.info(f"最大行動数: {args.max_actions}")
    logger.info(f"保存パス: {args.model_path}")
    logger.info(f"CUDA使用: {args.cuda}")
    logger.info("=" * 60)
    logger.info("")

    # 学習器を初期化
    learner = ActorCriticLearner(
        board_size=5,
        max_actions=args.max_actions,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        use_cuda=args.cuda,
    )

    # 既存モデルを読み込む場合
    if args.load_model:
        try:
            learner.network.load(args.load_model)
            logger.info(f"モデルを読み込みました: {args.load_model}")
            logger.info("")
        except FileNotFoundError:
            logger.warning(f"モデルファイル '{args.load_model}' が見つかりません")
            logger.info("新規に学習を開始します")
            logger.info("")

    # 学習を実行
    try:
        learner.train(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
        )
    except KeyboardInterrupt:
        logger.info("学習を中断しました")
        save = input("現在のモデルを保存しますか？ (y/n): ").strip().lower()
        if save in ["y", "yes"]:
            learner.network.save(args.model_path)
            logger.info("モデルを保存しました")

    # 学習進捗をグラフ化
    if args.plot:
        plot_training_progress(learner, "training_progress.png")

    logger.info("学習完了！")
    logger.info(f"モデルは {args.model_path} に保存されました")
    logger.info("次のコマンドでAIと対戦できます:")
    logger.info(f"  python3 play_with_ai.py --model {args.model_path}")


if __name__ == "__main__":
    main()
