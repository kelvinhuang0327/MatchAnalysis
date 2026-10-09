# 前一完整賽季固定得失分強度：一次訓練結果

此為獨立新候選 `prior_complete_season_net_runs_per_game_home_l2_logistic_v1`。舊季內累計候選保持未訓練。年度先後資料足夠後，本輪確實訓練 **一次**，沒有重跑近十場規則或搜尋參數。

| 同一批 102 場 | 命中 | 命中率 | Brier（較低較好） | logloss（較低較好） |
|---|---:|---:|---:|---:|
| 前季強度模型 | 57 | 55.88% | 0.246785 | 0.686703 |
| 同場市場 | 56 | 54.90% | 0.244466 | 0.682235 |
| 固定主隊方向／p=0.5 參考 | 57 | 55.88% | 0.250000 | 0.693147 |

2025 訓練 2,430/2,430 場，排除 0；2026 母體／同場比較 102/102，覆蓋 100%，排除 0。模型在這批 **102 場全部選主隊**，方向等同固定主隊。對市場雖多中 1 場（+0.9804 個百分點），Brier 較差 +0.00231940、logloss 較差 +0.00446768；因此沒有達成使用者接受的「同場準確率高於市場，且概率品質不惡化」。沒有據此改特徵、正則化、門檻或窗口，不宣稱改善已證實或策略獲利。

唯一特徵在看本輪成績前固定於 `protocol.json`：每隊前一完整 R 賽季的 `(得分總數－失分總數)/出賽數`，主隊值減客隊值；主場由不受罰截距表示。2024 全季彙總對應 2025 標籤學一次模型；2025 全季彙總固定用於整個 2026，沒有季內更新、先發回填或近十場變數。只有訓練案例學平均／母體標準差，2026 沿用該 scaler。固定 mean logloss＋½β²（L2 λ=1）、兩參數 damped Newton、0 初值、100 iteration／20 backtracking 上限、gradient tolerance 1e-8；沒有用其他 C/λ 縮放或 solver 替代。

本次 2 次 Newton 更新收斂，gradient infinity norm 3.7897e-9。截距 0.1717551673、標準化強度係數 0.0576365237，訓練 scale 1.071845287。係數小於主場截距在此批方向上的影響，導致全部推薦主隊；這是結果描述，不是另一次模型選擇。模型對市場的逐場命中分解為兩者都對 46、都錯 35、僅模型對 11、僅市場對 10；未作顯著性或泛化推論。

年度資料核對：2024 原始／獨立 CSV 各 2,429 場，2025 各 2,430 場，date/team/score multiset 完全一致；所有 CSV home_win 與比分一致。raw Retrosheet 的 date/gameNumber/away/home 唯一鍵沒有重複，各隊季內出賽序號從 1 至實際場數連續。2024 CLE、HOU 各 161 場，其餘 162；2025 全部 30 隊各 162。沒有把未進行的比賽補成第 162 場，也沒有把雙重賽只按日期／隊名去重。

2024 最晚 original/later-completion date 為 2024-09-30，比 2025 首場 2025-03-18 早 169 日；2025 最晚為 2025-09-28，比既有 2026 首場 2026-03-25 早 178 日。2024 的 3 場、2025 的 4 場跨日完成都已在下一季之前結束，因此各 physical game 只計一次。年度彙總與下一季間隔遠，沒有要求不必要的逐場 Final UTC。2026 目標本身若曾續賽，年度輸入不依賴其季內歷史；只要原開賽 identity 唯一、quote 在原 firstPitch 前、最終 label 一致，就能比較，故沒有沿用舊近十場視窗的七場排除。

隊伍完全按名稱及固定 MLB ID crosswalk，不作模糊配對。2024 OAK、2025 ATH 及已正常化 CSV 的 Athletics 都對应同一 franchise／MLB133；兩年度各有完整 30 隊。franchise 身份一致不代表陣容或球場不變；模型沒有調整搬遷、休季交易、場地或當季資訊。完整彙總及年度先後證據列於 `annual_source_audit.json`。

資料版本限定：2024 CSV metadata 的產生時間是 2026-05-24，晚於 2025 球季；2025 metadata 的驗證時間是 2026-03-12，但不等於當年 HTTP 收件或這個實驗模型的凍結時間。年度總表可能含賽後修訂，兩年曾有既有研究，2026 102 場賽果也已開封。因此全部是 **historical reconstruction development**，絕不稱當時可得、holdout 或前瞻。年度完結先後與精確檔案當年可得性是不同主張。

市場只重用既有 102 筆相同 snapshot／raw-record hash，按 gamePk、R、officialDate、gameNumber、主客 ID、scheduled start 唯一配對。雙邊價格重算比例無水位 p_home；quote 必須早於 source/official starts 與既有 actual firstPitch。沒有根據模型結果換價格、抓 closing price、宣稱決策 freshness、改最低 edge 或實際下注。固定主隊 p=0.5 是預先固定的無資訊概率參考，不能當主隊必勝。

單次真實 fit 已凍結於 `model_artifact.json`；`train_once.py` 偵測到它就立即拒絕，不能重訓或覆寫。日後重播只用保存係數，重新核來源 hash、年度彙總、訓練輸入 fingerprint、training-only scaler 與同場評分，不作第二次 fit：

```sh
python3.14 examples/prior_season_strength_v1/evaluate_saved.py > /tmp/match-prior-season-replay.json
python3.14 examples/prior_season_strength_v1/verify_replay.py
python3.14 examples/prior_season_strength_v1/test_research.py
```

JSON、CSV、annual audit 逐 byte 重播一致；保存係數的 objective／gradient 與固定 λ 獨立核對。六個聚焦測試涵蓋年度边界、train-only scaler、梯度／Hessian 有限差分、截距不受罰及 β 受罰、當前 target label 不影響保存模型概率、已存在模型拒絕第二次 fit 且 SHA 不變。最後一項的非零 exit 是預期拒絕，不是模型錯誤；沒有再擬合真實或合成資料。

`results.json` 保留全部 102 場概率、方向、比分標籤、排除及來源；`per_game.csv` 為逐場表。原 CSV/raw 與既有 repo capsule 沿用 hash-bound 檔案，不重複大量複製。搬遷舊 `data/mlb_2025` 來源時，給 `evaluate_saved.py`／`verify_replay.py` 一個該目錄參數即可；不同 hash 拒絕。沒有新 GET、付費 API、改憑證、安全設定、刪目錄、merge/push、監控或 ledger 寫入，Champion/main 及所有舊結果保持原樣。

Retrosheet attribution: “The information used here was obtained free of charge from and is copyrighted by Retrosheet.” More information: [Retrosheet](https://www.retrosheet.org/gamelogs/index.html), [欄位定義](https://www.retrosheet.org/gamelogs/glfields.txt).
