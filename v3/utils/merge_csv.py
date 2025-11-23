import pandas as pd
import numpy as np
import os

def merge_accuracy_data(base_path, output_filename):
    """
    讀取指定路徑下不同資料夾中的 csv 檔案，
    將 accuracy 欄位重命名，並將所有資料合併到一個新的 CSV 檔案中，
    且「不重新排序 class_name」。
    """
    merged_df = None
    class_order = None  # 保留最初讀到 class_name 的順序

    print("開始讀取並合併檔案...")
    
    for s in np.arange(0, 2.01, 0.1):
        s_str = f"s={s:.2f}"
        file_path = os.path.join(base_path, s_str, "class_1_to_100.csv")
        new_col_name = s_str

        try:
            df = pd.read_csv(file_path)

            if 'class_name' not in df.columns or 'accuracy' not in df.columns:
                print(f"警告：檔案 {file_path} 缺少必要的欄位 'class_name' 或 'accuracy'，已跳過。")
                continue

            df = df.rename(columns={'accuracy': new_col_name})
            df = df[['class_name', new_col_name]]

            # 記錄第一次讀到的 class_name 順序
            if merged_df is None:
                merged_df = df
                class_order = df['class_name'].tolist()
            else:
                merged_df = pd.merge(merged_df, df, on='class_name', how='outer')
                print(f"合併完成：{file_path}")

        except FileNotFoundError:
            print(f"警告：找不到檔案 {file_path}，已跳過。")
        except Exception as e:
            print(f"處理檔案 {file_path} 時發生錯誤：{e}")

    # 檢查是否有資料成功合併
    if merged_df is not None:

        # 🔥 關鍵：依照第一次的 class_name 順序重新排列
        merged_df = merged_df.set_index("class_name")
        merged_df = merged_df.reindex(class_order).reset_index()

        merged_df.to_csv(output_filename, index=False)
        print(f"\n成功！所有資料已合併並儲存至：{output_filename}")
    else:
        print("\n錯誤：沒有任何檔案成功讀取或合併。")
