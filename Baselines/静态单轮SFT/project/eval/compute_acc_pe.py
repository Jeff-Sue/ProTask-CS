import json
import matplotlib.pyplot as plt

data_path = "/home/aarc/CuhkszTeam/nas1/zefeng/RAG_agent/后处理_selected_compress_keywords/results/jieli_20to5_select_keywords.json"
golden_path = "golden.json"

# =========================
# load data
# =========================
with open(data_path, "r") as f:
    data = json.load(f)

with open(golden_path, "r") as f:
    golden_ids = json.load(f)

# =========================
# counters
# =========================
gold_top1 = 0
gold_top2_3 = 0
gold_top4_5 = 0

pred_top1 = 0
pred_top2_3 = 0
pred_top4_5 = 0

# =========================
# compute
# =========================
for idx, item in enumerate(data):

    ranked_list = item['retriever']['selected_case_ids'][:5]
    gold = golden_ids[idx]
    pred = item['policy']['case_id']

    # find rank position
    if gold in ranked_list:
        rank = ranked_list.index(gold)

        if rank == 0:
            gold_top1 += 1
            if pred == gold:
                pred_top1 += 1

        elif 1 <= rank <= 2:
            gold_top2_3 += 1
            if pred == gold:
                pred_top2_3 += 1

        elif 3 <= rank <= 4:
            gold_top4_5 += 1
            if pred == gold:
                pred_top4_5 += 1

# =========================
# plot data
# =========================
labels = ["Top1", "Top2-3", "Top4-5"]

gold_counts = [gold_top1, gold_top2_3, gold_top4_5]
pred_counts = [pred_top1, pred_top2_3, pred_top4_5]

x = range(len(labels))

plt.figure(figsize=(8, 5))

plt.bar([i - 0.2 for i in x], gold_counts, width=0.4, label="Golden Count")
plt.bar([i + 0.2 for i in x], pred_counts, width=0.4, label="Predicted Correct")

plt.xticks(list(x), labels)
plt.xlabel("Rank Bucket")
plt.ylabel("Count")
plt.title("PE: Golden vs Predicted Case Distribution (Top-5)")
plt.legend()

for i in x:
    plt.text(i - 0.2, gold_counts[i] + 0.3, str(gold_counts[i]), ha='center')
    plt.text(i + 0.2, pred_counts[i] + 0.3, str(pred_counts[i]), ha='center')

plt.tight_layout()

# =========================
# save figure
# =========================
plt.savefig('figures/golden_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n图表已保存为: golden_distribution_comparison.png")
