# 保守滯後當季勝率差：來源門檻未通過

本輪固定新增一個特徵：當季合格歷史主隊勝率減客隊勝率；歷史須至少早於目標 7 個曆日，每隊至少 20 場。對照保留前一完整年度得失分強度差與主場截距。依原始 2025 場次累計達 60% 的首個日期切分，同日不拆：截至 **2025-07-18 的 1,461 場訓練、之後 969 場評估**。來源與規則先寫入 `protocol.json`，没有查閱這個切段的模型成績、調參或重跑已開封的 102 場。

**尚未訓練兩個模型，沒有 accuracy/Brier/logloss 或係數貢獻結果。** 缺口是明確要求「所有延期場次排除」時所需的原定排程／改期證據，而不是逐場毫秒 UTC。

來源核對通過：2024/2025 raw 與獨立 CSV 的日期、隊伍、比分 multiset 相同；physical game key 含 date/gameNumber/away/home，沒有雙重賽錯配；隊伍出賽序號連續，2024 年度於 2025 首場前 169 日完結。2025 共 2,430 場，CSV 全為 Final/REG，raw 全有完整 game acquisition 標記。Retrosheet field14 記錄稍後日期完成的續賽／抗議補賽；4 場有該標記，另 2,426 場沒有稍後完成標記。

但 **blank field14 不能證明從未延期後改日開賽**；field161 的 Y 是已取得完整比賽資料，也不是原定排程證據。既有 as-played CSV 只有實際日期，沒有 original schedule、postponed 或 rescheduled-from 欄位；本機既有 2025 資料及專案 fixtures 未找到可配對來源。故所有 2,430 場延期狀態仍 unknown。這不表示 2,430 場真的延期，也不否認正常完整比賽在滯後日期前已結束；只是不能滿足「所有延期排除」這項額外條件。

| 原始分母 | 主要排除：已知跨日完成 | 主要排除：延期狀態未知 | 兩模型合格分母／覆蓋 |
|---|---:|---:|---:|
| 訓練 1,461 | 3 | 1,458 | 0／0% |
| 評估 969 | 1 | 968 | 0／0% |

非互斥原因計數另為延期未知 2,430、跨日完成 4。逐場 CSV/JSON 保留 identity、phase、來源欄位與原因；未知特徵為 null，不用 0 填補。baseline 與 candidate fit 都是 **0**；沒有為了填結果默認延期狀態為 false。待來源確認後，模型各至多 fit 一次、L2 λ=1、訓練資料標準化、同一合格分母及淘汰規則已固定；本 capsule 沒有提供繞過阻擋的訓練入口。

接續需要既有／獲授權、可按完整 game identity 配對的 2025 原定日期與改期證據；或使用者明確決定允許「開賽前曾延期，但在實際比賽日期已完賽」的歷史場次。尚未假設任何一項，也未新增賽事資料下載。即使之後模型通過，2025 已用過，只能標 historical reconstruction development，仍需要新資料及同場市場驗證，不能宣稱勝市場或獲利。

重播僅讀來源、重算門檻，不 fit、不寫 ledger：

```sh
python3.14 examples/lagged_season_winrate_v1/verify_replay.py
python3.14 examples/lagged_season_winrate_v1/test_source_gate.py
python3.14 examples/lagged_season_winrate_v1/source_gate.py > /tmp/match-lagged-gate-replay.json
```

來源沿用 `/Users/kelvin/Kelvin-WorkSpace/Betting-pool/data/mlb_2025` 的 6 個原檔，雜湊固定於 protocol；搬遷可給 source_gate.py／verify_replay.py 一個該目錄參數，不同 hash 拒絕。重播逐 byte 核對 JSON/CSV、capsule 與封存依賴。5 個聚焦合成測試驗證同日不拆、切點不依賴 labels、未知延期不默認 false、續賽與不完整來源排除、7 日邊界／未来／target label 不進特徵、最低 20 場不足不插補。

主線、Champion、ledger、原 102 場及先前模型都未改；沒有 merge/push、下注、刪檔、付費 API 或重型實驗。

Retrosheet attribution: “The information used here was obtained free of charge from and is copyrighted by Retrosheet.” [欄位定義](https://www.retrosheet.org/gamelogs/glfields.txt)／[Game Logs](https://www.retrosheet.org/gamelogs/index.html)。本輪網頁查閱只確認欄位定義，沒有下載新賽事資料。
