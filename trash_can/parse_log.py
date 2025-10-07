import re
import csv

logfile = "uap_outputs/lr=0.01_30_out.log"
outfile = "results.csv"

pattern_loss = re.compile(r"avg \(-loss\) update: (-?\d+\.\d+)")
pattern_acc = re.compile(r"Acc subset(\d) : (\d+\.\d+)%")

results = []

with open(logfile, "r") as f:
    lines = f.readlines()

loss = None
accs = {}

for line in lines:
    # 找 loss
    m_loss = pattern_loss.search(line)
    if m_loss:
        loss = float(m_loss.group(1))
        accs = {}  # reset accs
        continue

    # 找 acc
    m_acc = pattern_acc.search(line)
    if m_acc:
        subset_id = int(m_acc.group(1))
        acc_value = float(m_acc.group(2))
        accs[subset_id] = acc_value

        # 如果三個 acc 都有了，記錄一筆結果
        if len(accs) == 3 and loss is not None:
            results.append([loss, accs[1], accs[2], accs[3]])
            loss = None  # reset

# 寫 CSV
with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["loss", "acc1", "acc2", "acc3"])
    writer.writerows(results)

print(f"Saved {len(results)} rows to {outfile}")
