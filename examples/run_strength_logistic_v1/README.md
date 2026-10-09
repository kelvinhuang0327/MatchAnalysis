# 得失分強度＋主場 logistic：資料閘門交付

本輪依使用者要求，在真正完成時間缺失時停止。唯一候選已固定，但**沒有實作／執行擬合、沒有候選預測或新成績**；不是模型失敗，而是資料條件未成立。既有較早完整賽季的比分與標籤可核對，不能把日期或顯示開賽時刻補成實際完成 UTC。

| 範圍 | 原始分母／可核標籤 | 實際完成 UTC 來源 | 本輪處理 |
|---|---:|---:|---|
| 2025 首選訓練賽季 | 2,430／2,430 | 0／2,430 | 訓練特徵合法入選 0，擬合 0 |
| 2024 更早替代賽季 | 2,429／2,429 | 0／2,429 | 相同時間缺口，不替换訓練範圍 |
| 2026 既有已開封評估母體 | 102／102 | 季累計歷史不完整 | 同場模型比較 0，102 場皆未入候選比較 |

2025 的 display EDT start time 2,430 筆均非空；這不等於帶證據的實際 firstPitch／completed_at UTC。兩年 raw game logs 都有分鐘時長，但 2024 有 3 筆、2025 有 4 筆跨日完成標記；不能把原開賽日期當完成日期。[Retrosheet 官方欄位表](https://www.retrosheet.org/gamelogs/glfields.txt)明定第 14 欄是稍後完成的日期及中斷資訊，第 19 欄是分鐘時長，未提供完成 UTC 時刻。CSV 又省略 game number：2024 有 29、2025 有 28 筆 date/team 重複超額；raw 的 date/gameNumber/away/home 鍵則分別有 2,429、2,430 個唯一值，續用必須保留它，不能只按日期及隊名配對。

2026 的 102 筆雙邊價格及已開封賽果都可唯一配對，quote 早於 source/official scheduled start 和既有官方 firstPitch；價格不是主要阻礙。新候選採與該固定 quote 相同的特徵截止，不能使用後來賽事知識對比更早市場。季累計特徵涉及 1,151 個潛在先前 scored R gamePk；既有已查證 archived Final 子集只覆蓋 64，另 1,087 不在此子集中。64 場子集也不是完整 30 隊季累計時間來源；沒有藉此拼湊不完整強度。這些數字是資料覆蓋核算，不是模型樣本或新增成績。

`protocol.json` 在任何候選成績之前固定唯一設定，後續成效優先順序保存於 `interpretation_policy.json`，沒有因此改規則或開第二實驗：

- 同季至截止前全部已完成 R，以每隊至少 20 場暖機，不搜尋視窗。
- x1＝主隊平均得分－客隊平均得分；x2＝客隊平均失分－主隊平均失分。主場用不受罰截距，沒有重複常數特徵。
- 只用訓練集平均與母體標準差標準化；固定 objective＝mean binary logloss＋½(β1²＋β2²)，L2 λ=1，截距不受罰。未用 sklearn C 的不同縮放定義替代。
- 固定最多一次三參數 damped Newton 擬合，100 iterations、20 backtracking、gradient infinity-norm tolerance 1e-8；不收斂就失敗，不換 solver 或調參。本輪此 solver 尚未實作／執行。
- 優先完整 2025 訓練、2026-06-14 至 06-22 既有 102 場 development 評估；2024/2025/2026 都曾用於既有研究，不稱未見 holdout。
- 缺實際完成來源、完整季累計或無法解析續賽，就拒絕 target，不補值、不忽略未知歷史。雙重賽必須帶 game number；續賽不得重複計 physical game。
- 直接重用既有固定雙邊 pregame quote／raw-record hash，不按結果換快照、不拿 closing price 替代；研究 quote 年齡只描述，正式風控不改。
- 模型及市場分類以 p_home≥0.5 選主隊；固定主隊方向 always HOME，其概率控件預先固定 p_home=0.5，明示為無資訊參考，不能當主隊必勝。

使用者接受的成效優先順序是：同場合格資料準確率高於市場，概率品質不惡化；未提供額外數字門檻。本輪同場分母為 0，候選／市場／固定主隊的 accuracy、Brier、logloss 都為 null，成效標 `NOT_EVALUABLE`。沒有把固定方法或資料閘門通過當作準確率提升。

最小待補資料是：所用訓練 target 的唯一完整身份／實際開賽 UTC，以及所有可能進入訓練與評估強度累計的先前比賽之**實際 Final 事件 UTC、原始來源 SHA 和續賽完成證據**。`strict_completion_reason` 會拒絕 firstPitch＋duration 推算、date-only、沒有 timezone、未 Final、缺原始 hash、未解析續賽及完成時間不早於 cutoff 的資料。`data_gate_result.json` 列出 102 場逐場缺口與 1,087 個未覆蓋的 gamePk。這不是授權下載這些資料；本輪新賽事資料 GET=0，只有唯讀官方格式文件查證。

可行範圍仍是既有 2025／2024 比分標籤和 102 筆市場來源；補齊時間證據後才可落實此單一候選。date-only／前一日／分鐘時長重建會放寬本次明定條件，沒有自行採用。沒有重跑已淘汰近十場規則、搜尋超參數、大量下載或製造未見樣本。

只讀重播及聚焦測試：

```sh
python3.14 examples/run_strength_logistic_v1/data_gate.py > /tmp/match-run-strength-gate.json
python3.14 examples/run_strength_logistic_v1/verify_replay.py
python3.14 examples/run_strength_logistic_v1/test_data_gate.py
```

重播要求 `protocol.json` 指定的舊 CSV/raw/metadata/research summary 都符合 SHA；repo 內來源按當前 checkout 解析。舊資料若在另一個位置，可帶 capsule path 與 legacy `data/mlb_2025` 路徑：

```sh
python3.14 examples/run_strength_logistic_v1/data_gate.py examples/run_strength_logistic_v1 /path/to/legacy/data/mlb_2025
python3.14 examples/run_strength_logistic_v1/verify_replay.py /path/to/legacy/data/mlb_2025
```

舊 raw 不重複複製進 bundle；外部來源路徑及 SHA 都明列在 protocol，缺檔或不同 hash 時拒絕重播。JSON 逐 byte 一致；六個合成聚焦測試驗證實際事件、推算拒絕、截止邊界、續賽、Final/hash、比分標籤與雙重賽身份。只新增這個資料閘門 capsule，沒有修改 Champion、main、ledger、frozen predictions 或任何舊封存結果，也沒有 merge/push、下單、監控或安全設定變更。

Retrosheet attribution: “The information used here was obtained free of charge from and is copyrighted by Retrosheet.” More information: [Retrosheet](https://www.retrosheet.org/gamelogs/index.html).
