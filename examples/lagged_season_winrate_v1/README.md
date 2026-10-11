# 保守滯後當季勝率差：一次 development 實驗

前季基準與新增一個當季勝率差的候選，各實際 fit **一次**；沒有參數搜尋或評估後調整。共同評估 968 場，候選通過预定淘汰規則，但這是已用過的 2025 歷史重建 development，沒有新 holdout、前瞻或同場市場驗證。

| 同一批 968 場 | 命中 | accuracy | Brier（低較好） | logloss（低較好） |
|---|---:|---:|---:|---:|
| 前季強度基準 | 524 | 54.1322% | 0.247424 | 0.687979 |
| 加當季勝率差 | 527 | 54.4421% | 0.247000 | 0.687126 |
| 固定主隊／p=0.5 參考 | 516 | 53.3058% | 0.250000 | 0.693147 |

候選比基準多對 3 場（+0.3099 個百分點），Brier −0.000423338、logloss −0.000853063；两者都對 506、都錯 423、候選獨對 21、基準獨對 18。沒有作顯著性、泛化或獲利推論，更未達成或宣稱使用者的勝市場目標。

來源與完成日期：完整 161 欄 Retrosheet final game log、field161=Y、獨立 CSV Final/REG 與日期／隊伍／比分 multiset 一致，共同支持實際日期與已完成比賽。官方 field14 定義稍後日期完成的續賽／抗議補賽；4 場有该標記，均從當季歷史及目標排除；其餘無較晚完成標記且有上述完整 final 證據。blank14 單獨不是完成證據。這是來源的曆日解析，沒有推造 UTC 完成時戳或當年版本可得性。依明確方法修正，開賽前曾延期不影響實際已完賽日期，不再要求原定排程；修正前 protocol hash 保留於新 protocol。固定切點、門檻、特徵與 λ 均未重調。

按原始日期／場數累計達 60% 的首日切分，同日不拆：截至 **2025-07-18 的 1,461 場訓練、之後 969 場評估**。train/evaluation 比賽日期完全分離。4 場續賽排除後不重新切點。

| 階段 | 原始分母 | 跨日完成排除 | 20 場門檻不足排除 | 兩模型共同合格／覆蓋 |
|---|---:|---:|---:|---:|
| 訓練 | 1,461 | 3 | 382 | 1,076／73.6482% |
| 評估 | 969 | 1 | 0 | 968／99.8968% |

全部 2,426 場正常完整來源可作當季歷史，包括尚未達目標暖機門檻的早期場次。每個目標只使用完成來源日期 <= 目標日期減 7 曆日的歷史，每隊至少 20 場；主隊 wins/games 減客隊 wins/games 為唯一新增特徵。不平滑、不換窗口、不以缺值補 0。評估期間較早已完成且滿足 7 日滯後的賽果可更新後續特徵，模型係數／scaler 不更新；當前或未來賽果不入自身特徵。全部歷史比分曾開封，仍只稱 reconstruction development。

基準為 2024 全季 `(得分−失分)/出賽數` 的主客差，加不受罰主場截距；全季以固定值用於 2025。2024/2025 日期、隊伍、比分與 raw/CSV multiset 一致；physical key 含 date/gameNumber/away/home，出賽序號連續，前季最晚完成比 2025 首場早 169 日。沿用精確30隊 crosswalk（OAK/ATH 同 franchise），没有 fuzzy 配對。這次基準在共同 1,076 訓練案例擬合，並非重用既有全年度模型係數；舊模型保持封存。

兩模型皆使用訓練共同案例的標準化（各欄 mean/population SD），固定 objective 為 mean logloss + 0.5*sum(beta²)，L2 λ=1，截距不受罰。固定 2/3 參數 damped Newton，0 初值、100 iter／20 backtracking、gradient <=1e-8，无 solver 替代。兩者均 2 次更新收斂：基準 gradient 2.6674e-10，候選 8.2310e-10。

| 模型 | 截距 | 標準化前季係數 | 標準化當季係數 |
|---|---:|---:|---:|
| 基準 | 0.1079860873 | 0.0530905174 | — |
| 候選 | 0.1080763914 | 0.0476238731 | 0.0380187244 |

候選評估 logit 貢獻的平均絕對值：截距 **0.10808**、前季 **0.03849**、當季 **0.02064**。主場截距絕對值大於兩訊號絕對貢獻合計的場次為 845/968；候選方向 903 主／65 客，基準 942 主／26 客。新增訊號讓部分場次改選客隊，增量仍小且主場截距多數主導；不能據此調小正則化或改門檻。逐場概率與三項 logit 貢獻在 `per_game_gate.csv`，總結在沿用的 `source_gate_result.json`，不新增純 gate 報表。

`model_artifact.json` 先以 exclusive create 保留執行權，再於各 fit 前紀錄 attempt=1；若檔案已存在立即拒絕，因此不會第二次 fit 或覆寫模型。今後只用保存係數重播：

```sh
python3.14 examples/lagged_season_winrate_v1/research.py > /tmp/match-lagged-replay.json
python3.14 examples/lagged_season_winrate_v1/verify_replay.py
python3.14 examples/lagged_season_winrate_v1/test_source_gate.py
python3.14 examples/lagged_season_winrate_v1/test_research.py
```

10 項聚焦合成測試涵蓋同日切分、切點與 label 無關、完整 final 證據、跨日排除、7 日／未來邊界、歷史不足、梯度與 Hessian 有限差分、L2／截距處罰、與封存基準的 objective 一致、train-only scaler、target label 不影響保存概率、existing artifact 拒絕重訓且 hash 不變。測試不 fit；重播核 JSON/CSV 逐 byte 相同、來源／訓練指紋／scaler／程式版本、保存係數 objective／gradient，只算一次導數、不重訓。

6 個原始來源檔沿用 legacy `data/mlb_2025` 並鎖 hash；搬遷可給 research.py／verify_replay.py 一個該資料目錄參數。歷史來源可能賽後修訂且已用過，當年 snapshot 可得性未證明。沒有新賽事 GET、付費 API、ledger 寫入、下注、刪檔、merge/push；main、Champion、102 場和舊 capsule 未改。下一階段需新資料及合法同場賽前雙邊價格；本輪不開始監控或虛構市場比較。

Retrosheet attribution: “The information used here was obtained free of charge from and is copyrighted by Retrosheet.” [欄位定義](https://www.retrosheet.org/gamelogs/glfields.txt)／[Game Logs](https://www.retrosheet.org/gamelogs/index.html)。
