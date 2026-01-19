import os
import git
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import jieba
from wordcloud import WordCloud
from dotenv import load_dotenv
import time  # 新增：控制线程速度，避免冲突

# 加载环境变量（可选）
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# 配置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False


# 修复Git权限问题（自动执行安全目录配置）
def fix_git_ownership(repo_path):
    """自动配置Git安全目录，解决dubious ownership报错"""
    repo_path = repo_path.replace("\\", "/")  # 统一路径分隔符
    try:
        git.cmd.Git().execute(f"git config --global --add safe.directory {repo_path}")
        print(f"已配置Git安全目录：{repo_path}")
    except Exception as e:
        print(f"配置Git安全目录失败（不影响运行）：{str(e)}")


class GitHubCommitAnalyzer:
    def __init__(self, repo_url, local_repo_path="./temp_repo"):
        self.repo_url = repo_url
        self.local_repo_path = local_repo_path
        self.repo = None
        self.commits_df = None

    def clone_repo(self):
        # 先修复Git权限
        fix_git_ownership(os.path.abspath(self.local_repo_path))

        if not os.path.exists(self.local_repo_path):
            print(f"正在克隆仓库: {self.repo_url}")
            if GITHUB_TOKEN:
                repo_url_with_token = self.repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")
                self.repo = git.Repo.clone_from(repo_url_with_token, self.local_repo_path)
            else:
                self.repo = git.Repo.clone_from(self.repo_url, self.local_repo_path)
            print("仓库克隆完成")
        else:
            self.repo = git.Repo(self.local_repo_path)
            print("本地仓库已存在，直接加载")

    def extract_commit_data(self):
        """稳定版：单线程+限速，避免资源冲突"""
        print("正在提取提交数据（稳定模式）...")
        # 1. 限制提交数量，保证速度+稳定性
        commits = list(self.repo.iter_commits(max_count=800))
        commit_data = []

        # 2. 单线程解析（避免多线程冲突），加微小延时
        for idx, commit in enumerate(commits):
            try:
                # 每解析20条提交，暂停0.1秒，避免IO过载
                if idx % 20 == 0:
                    time.sleep(0.1)

                commit_info = {
                    "commit_hash": commit.hexsha[:8],
                    "author_name": commit.author.name,
                    "author_email": commit.author.email,
                    "commit_time": datetime.fromtimestamp(commit.committed_date),
                    "commit_message": commit.message.strip(),
                    "insertions": commit.stats.total["insertions"],
                    "deletions": commit.stats.total["deletions"]
                }
                commit_data.append(commit_info)
            except Exception as e:
                print(f"跳过异常提交{commit.hexsha[:8]}：{str(e)[:50]}")  # 简化报错输出
                continue

        # 3. 转换为DataFrame并保存
        self.commits_df = pd.DataFrame(commit_data)
        self.commits_df = self.commits_df.sort_values("commit_time").reset_index(drop=True)

        if not os.path.exists("data"):
            os.makedirs("data")
        self.commits_df.to_csv("data/commit_history.csv", index=False, encoding="utf-8-sig")
        print(f"数据提取完成，共{len(self.commits_df)}条有效提交记录！")

    def analyze_contributor_distribution(self):
        """贡献者分析（保留原功能）"""
        print("\n=== 贡献者提交次数分析 ===")
        contributor_stats = self.commits_df["author_name"].value_counts().head(10)
        print(contributor_stats)

        plt.figure(figsize=(12, 6))
        contributor_stats.plot(kind="barh", color="#1f77b4")
        plt.title("Top10 贡献者提交次数", fontsize=14)
        plt.xlabel("提交次数", fontsize=12)
        plt.ylabel("贡献者姓名", fontsize=12)
        plt.tight_layout()
        plt.savefig("data/contributor_top10.png", dpi=300)
        print("贡献者图表已保存")

    def analyze_commit_trend(self, freq="ME"):  # 修复：M→ME
        """提交趋势分析（消除pandas警告）"""
        print("\n=== 提交频率趋势分析 ===")
        self.commits_df["commit_date"] = self.commits_df["commit_time"].dt.date
        # 修复：用ME替代M，W保持不变（W-SUN是默认值）
        commit_trend = self.commits_df.groupby(pd.Grouper(key="commit_time", freq=freq)).size()
        print(f"按{freq}统计（最近10条）：")
        print(commit_trend.tail(10))

        plt.figure(figsize=(14, 6))
        commit_trend.plot(kind="line", color="#ff7f0e", linewidth=2)
        freq_label = {"ME": "月", "W": "周", "D": "日"}[freq]  # 适配ME
        plt.title(f"提交频率趋势（按{freq_label}统计）", fontsize=14)
        plt.xlabel("时间", fontsize=12)
        plt.ylabel("提交次数", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"data/commit_trend_{freq}.png", dpi=300)
        print(f"趋势图表已保存")

    def analyze_code_churn(self):
        """代码变动分析（保留原功能）"""
        print("\n=== 代码变动量分析 ===")
        self.commits_df["commit_year"] = self.commits_df["commit_time"].dt.year
        code_churn = self.commits_df.groupby("commit_year").agg({
            "insertions": "sum",
            "deletions": "sum"
        }).round(2)
        code_churn["net_change"] = code_churn["insertions"] - code_churn["deletions"]
        print("按年统计代码变动：")
        print(code_churn)

        plt.figure(figsize=(14, 6))
        code_churn[["insertions", "deletions"]].plot(kind="bar", stacked=True,
                                                     color=["#2ca02c", "#d62728"])
        plt.title("按年代码变动量（新增/删除行数）", fontsize=14)
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("代码行数", fontsize=12)
        plt.legend(["新增行数", "删除行数"])
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("data/code_churn_year.png", dpi=300)
        print("代码变动图表已保存")

    def generate_commit_message_wordcloud(self):
        """词云分析（保留原功能）"""
        print("\n=== 提交信息词云分析 ===")
        all_messages = " ".join(self.commits_df["commit_message"].dropna())
        words = jieba.lcut(all_messages)
        stop_words = {"the", "a", "an", "and", "or", "to", "in", "for", "of", "with", "on", "at", "by", "is", "are",
                      "was", "were", "I", "we", "this", "that", "it", "fix", "fixed", "update", "add", "remove",
                      "clean", "up"}
        filtered_words = [word for word in words if len(word) >= 2 and word.lower() not in stop_words]
        word_text = " ".join(filtered_words)

        wordcloud = WordCloud(
            width=1200, height=600,
            background_color="white",
            font_path="simhei.ttf" if os.name == "nt" else "Arial",
            max_words=200,
            colormap="viridis"
        ).generate(word_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("提交信息词云", fontsize=16)
        plt.tight_layout()
        plt.savefig("data/commit_message_wordcloud.png", dpi=300, bbox_inches="tight")
        print("词云图表已保存")

    def run_full_analysis(self):
        self.clone_repo()
        self.extract_commit_data()
        self.analyze_contributor_distribution()
        self.analyze_commit_trend(freq="ME")  # 按月统计（ME）
        self.analyze_commit_trend(freq="W")  # 按周统计
        self.analyze_code_churn()
        self.generate_commit_message_wordcloud()
        print("\n=== 所有分析完成！结果保存在data目录 ===")


if __name__ == "__main__":
    TARGET_REPO = "https://github.com/psf/requests.git"
    analyzer = GitHubCommitAnalyzer(repo_url=TARGET_REPO)
    analyzer.run_full_analysis()