import pandas as pd
import numpy as np
import os

def merge_accuracy_data(base_path="output/addition", output_filename="merged_accuracies.csv"):
    """
    讀取指定路徑下不同 alpha 值資料夾中的 class_1_to_10.csv 檔案，
    將 accuracy 欄位重命名為對應的 alpha 值，並將所有資料合併到一個新的 CSV 檔案中。

    Args:
        base_path (str): 包含 alpha=* 資料夾的基礎路徑。
        output_filename (str): 輸出 CSV 檔案的名稱。
    """
    alpha_values = np.arange(0, 1.01, 0.05)
    merged_df = None

    print("開始讀取並合併檔案...")
    
    # 迭代每個 alpha 值
    for alpha in alpha_values:
        alpha_str = f"{alpha:.2f}"
        file_path = os.path.join(base_path, f"alpha={alpha_str}", "class_1_to_10.csv")
        # file_path = os.path.join(base_path, f"alpha=-{alpha_str}", "class_1_to_10.csv")
        new_col_name = alpha_str
        try:
            df = pd.read_csv(file_path)
            if 'class_name' not in df.columns or 'accuracy' not in df.columns:
                print(f"警告：檔案 {file_path} 缺少必要的欄位 'class_name' 或 'accuracy'，已跳過。")
                continue
            df = df.rename(columns={'accuracy': new_col_name})
            df = df[['class_name', new_col_name]]
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='class_name', how='outer')
                print(f"合併完成：{file_path}")

        except FileNotFoundError:
            print(f"警告：找不到檔案 {file_path}，已跳過。")
        except Exception as e:
            print(f"處理檔案 {file_path} 時發生錯誤：{e}")
    
    # 檢查是否有資料成功合併
    if merged_df is not None:
        merged_df.to_csv(output_filename, index=False)
        print(f"\n成功！所有資料已合併並儲存至：{output_filename}")
    else:
        print("\n錯誤：沒有任何檔案成功讀取或合併。")


merge_accuracy_data(base_path="output/addition", output_filename="output/addition/addition.csv")
# merge_accuracy_data(base_path="output/addition", output_filename="output/addition/subtraction.csv")