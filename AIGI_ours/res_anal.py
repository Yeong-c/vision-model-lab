import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve

def aggregate_results_with_thresholds(root_dir='results', output_file='prediction_all.txt'):
    all_summaries = []
    
    # 1. 하위 폴더 순회 (eval_로 시작하는 폴더 대상)
    if not os.path.exists(root_dir):
        print(f"❌ Error: '{root_dir}' 폴더를 찾을 수 없습니다.")
        return
        
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f.startswith('eval_')]
    folders.sort()
    
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        summary_path = os.path.join(folder_path, 'summary.txt')
        preds_path = os.path.join(folder_path, 'predictions.csv')
        
        if not (os.path.exists(summary_path) and os.path.exists(preds_path)):
            continue
            
        # 2. summary.txt 파싱 (체크포인트 정보 포함)
        summary_info = {}
        with open(summary_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    key, val = line.split(':', 1)
                    summary_info[key.strip()] = val.strip()
        
        # 3. predictions.csv 로드 및 분석
        df = pd.read_csv(preds_path)
        y_true = df['label'].values
        y_score = df['score'].values  # Logit values
        
        # --- [Standard: Threshold = 0] ---
        y_pred_std = (y_score > 0).astype(int)
        acc_real = accuracy_score(y_true[y_true == 0], y_pred_std[y_true == 0]) if any(y_true == 0) else 0
        acc_fake = accuracy_score(y_true[y_true == 1], y_pred_std[y_true == 1]) if any(y_true == 1) else 0
        balanced_acc_std = (acc_real + acc_fake) / 2
        
        # --- [Advanced: Find Best Threshold] ---
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        best_tau = 0.0
        best_acc = 0.0
        
        for t in thresholds:
            y_pred_temp = (y_score > t).astype(int)
            cur_real = accuracy_score(y_true[y_true == 0], y_pred_temp[y_true == 0]) if any(y_true == 0) else 0
            cur_fake = accuracy_score(y_true[y_true == 1], y_pred_temp[y_true == 1]) if any(y_true == 1) else 0
            cur_balanced = (cur_real + cur_fake) / 2
            
            if cur_balanced > best_acc:
                best_acc = cur_balanced
                best_tau = t

        # 결과 데이터 구조화
        summary_info['Folder Name'] = folder
        summary_info['Standard Balanced Acc'] = f"{balanced_acc_std:.4f}"
        summary_info['Best Threshold (Logit)'] = f"{best_tau:.4f}"
        summary_info['Best Balanced Acc'] = f"{best_acc:.4f}"
        summary_info['Acc Improvement'] = f"{best_acc - balanced_acc_std:.4f}"
        summary_info['Real Acc (@Std)'] = f"{acc_real:.4f}"
        summary_info['Fake Acc (@Std)'] = f"{acc_fake:.4f}"
        
        all_summaries.append(summary_info)

    # 4. 통합 리포트 생성 (Checkpoint 기록 추가)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*95 + "\n")
        f.write(f"{'LEDD INTEGRATED PERFORMANCE REPORT (ACCURACY & THRESHOLDS)':^95}\n")
        f.write("="*95 + "\n\n")
        
        for item in all_summaries:
            f.write(f"▶ Folder: {item['Folder Name']}\n")
            f.write(f"   - Dataset         : {item.get('Dataset', 'N/A')}\n")
            # [추가된 부분] 사용된 체크포인트 파일명 기록
            f.write(f"   - Checkpoint      : {item.get('Checkpoint', 'N/A')}\n")
            f.write(f"   - ROC-AUC         : {item.get('ROC-AUC', 'N/A')}\n")
            f.write("-" * 55 + "\n")
            f.write(f"   [Standard Metric (Logit 0)]\n")
            f.write(f"   - Balanced Acc    : {item['Standard Balanced Acc']}\n")
            f.write(f"   - Real/Fake Acc   : {item['Real Acc (@Std)']} / {item['Fake Acc (@Std)']}\n")
            f.write("-" * 55 + "\n")
            f.write(f"   [Optimal Metric (Calibration)]\n")
            f.write(f"   - Best Threshold  : {item['Best Threshold (Logit)']}\n")
            f.write(f"   - Best Balanced Acc: {item['Best Balanced Acc']}\n")
            f.write(f"   - Potential Gain  : +{item['Acc Improvement']}\n")
            f.write("\n" + "."*95 + "\n\n")

    print(f"✅ 분석 완료! '{output_file}'에 체크포인트 정보와 임계값 분석 결과가 저장되었습니다.")

if __name__ == "__main__":
    aggregate_results_with_thresholds()