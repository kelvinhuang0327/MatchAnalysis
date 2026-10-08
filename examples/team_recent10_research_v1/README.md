# 球隊近十場研究基準：一次固定方向規則

母體是原 archive 已開封的 2026-06-14 至 2026-06-22 全部 102 場價格合格 R 比賽；完整保留分母，沒有按賽果挑場。這是獨立探索基準，沒有恢復或修改 Champion，也不是新 holdout、前瞻樣本或方法獲利證明。

固定規則與來源 SHA 已在 `protocol.json` 中保存，早於本輪補查及評分：兩隊各取目標 scheduled start 前已完成的最近十場 R，按 scheduled start 排序，最低十場。勝場較多隊作方向推薦；相同則 `TIE_ABSTAIN`。沒有把十場勝率映射為勝負概率；Brier、log loss、ROI 都保留 null。市場控件使用原雙邊價格的比例無水位概率，取最高概率隊；固定主隊控件一律選主隊。

| 分母／規則 | 有方向場次 | 命中 | 命中率 |
|---|---:|---:|---:|
| 同一批 77 場：近十場方向 | 77 | 31 | 40.26% |
| 同一批 77 場：市場方向 | 77 | 39 | 50.65% |
| 同一批 77 場：固定主隊 | 77 | 40 | 51.95% |
| 全部可用 95 場：市場方向 | 95 | 53 | 55.79% |
| 全部可用 95 場：固定主隊 | 95 | 52 | 54.74% |

102 場中，95 場可唯一配對且兩隊歷史視窗可重建；18/95 場十場勝率平手，不推薦方向，另列在結果中。近十場規則的有方向覆蓋率為 77/95=81.05%，以完整母體則為 77/102=75.49%。平手不當命中或失敗，也没有另組混合規則。7/102 場排除於比較之前，原因如下。

官方 schedule 的 gamePk 824912 有 6/16 原開賽與 6/17 續賽兩個表示，firstPitch 卻仍是原開賽時間。若直接以原 firstPitch 加時長，會錯把續賽前當已完成。因此只計同一 physical game 一次，且在這場可能進入近十場的視窗中停止計算。受影響排除 gamePk：824913、823853、824910、823854、824909、824908、823852；沒有補猜結束時間或偷偷移除這場再拿更早賽事填滿。完整排除理由及候選來源保留於 `results.json`。

本次近十場規則在同場 77 場少於市場 8 場、少於固定主隊 9 場；與市場逐場比較為兩者都對 19、兩者都錯 26、僅近十場對 12、僅市場對 20。這些都是已看結果的描述，沒有顯著性、泛化改善或獲利結論，也未據此反轉規則、搜尋參數或修改風控。

資料處理只補兩個免費官方 schedule GET：一個單日欄位能力查證、一個必要 gamePk 欄位投影，共 93,137 bytes，正常 TLS。後者涵蓋 276 個必要 gamePk、277 個表示（包含上述續賽重複），只補 identity、status、firstPitch 與 gameDurationMinutes；沒有下載先發、Champion inputs 或批量歷史 timecode。原 cache 提供比分，原已開封結果提供 target outcome，兩者逐場一致才可比較。`source_captures.json` 保存真正 request/receipt/persist UTC、URL 與 SHA；收件為 2026-10-08，不能回填成 6 月已知資訊。

完成界線使用官方歷史 firstPitch 加 gameDurationMinutes，再加 60 秒容許分鐘取整；只納入該界線嚴格早於 target scheduled start 的 Final R。這是歷史完成時間重建，條件是該 duration 能代表一般比賽經過時長，並非逐場實際 last-out 或 contemporaneous Final 版本證明。明確的暫停／續賽則排除。納入歷史的最短界線餘裕為 49,860 秒（13 小時 51 分）；没有使用 target 同 officialDate 的歷史比賽。賽程集合、比分及時長都在目標賽後取得，後續修訂及完整集合的時點可用性仍無法認證。沒有把 Preview 的賽後 record 當特徵，也沒有回填實際先發。

`history_locators.json` 只按既有 cache 定位每隊最近十一個候選，供一個可能尚未完成的 prior-start 比賽讓位；不使用結果優化 buffer。physical game 去重後逐一檢查完成順序；不明的近期完成狀態會使 target 排除，不能略過後當作完整十場。原 102 場唯一配對保留，7 場排除均出自此續賽順序缺口。

輸出：`results.json` 包含全部 102 場、逐場所用歷史、completion 證據、排除、平手、同場控件及每日分母；`per_game.csv` 是可直接檢視的逐場表。離線重跑不發任何網路請求：

```sh
python3.14 examples/team_recent10_research_v1/evaluate.py > /tmp/match-recent10-replay.json
python3.14 examples/team_recent10_research_v1/verify_replay.py
python3.14 examples/team_recent10_research_v1/test_research.py
```

source manifest 綁定輸入、固定規則、raw 與輸出；raw schedule 沿用 repo 既有 fixture，SHA `6950ed55ff1ddeb9b90f90138847ed7ebe79ce39d37614268d344af96eb65e55`。JSON／CSV 重播逐 byte 相同。六個聚焦檢查涵蓋平手不建立概率、每隊十場且無重複／未來／自身賽果、續賽拒絕、缺完成時間不能無聲補值、改 target 賽果不影響自身方向、明確未完成賽事不能進入視窗；所有測試修改僅在隔離合成副本。

只交付此 capsule；主線、原研究 capsule、Champion、正式 ledger/frozen predictions、資料庫及既有風控都未修改。無 merge/push、監控、真實下單、付費 API 或新方法搜尋。
