# 兩場固定報價截止的 Champion 特徵恢復

只處理 824673、824269；原失敗目標 823457 未再查。重用既有官方 cache 定位，再取必要的免費官方歷史 feed 小型欄位投影。没有預測、方法評分、調參、新假說、ledger 寫入或監控。

| 目標／固定 TSL quote UTC | 已恢復欄位 | 完整輸入可行性 |
|---|---|---|
| 824673／2026-06-05 18:18:59.881082 | target Preview 版本 17:42:15；Ray 12 次先發 ER=31、outs=188，ERA=837/188=4.452127659…；Cabrera 10 次先發 ER=24、outs=162，ERA=4；CHC 十場 4 勝、SF 十場 3 勝，差值 0.1。所用逐場分數、登板統計都有報價前 Final 版本證據。 | 現有官方索引內的兩項特徵數值可重建；完整歷史輸入認證仍不確定。索引取得於賽後，沒有完整賽事集合的賽前版本，不能證明沒有遺漏或後續修改。 |
| 824269／2026-06-06 15:36:02.469277 | target Preview 版本 14:23:51；Miller 4 次登板逐場核對，僅計 3 次先發 ER=2、outs=48，ERA=54/48=1.125；排除後援 ER=2、outs=15。DET 十場 5 勝。 | 本輪不可完整恢復：SEA 的 823135 沒有取得 Final 回應；Montero 的 824290 回傳 Live，完整先發 ERA 停止恢復。不用季 ERA 補值。 |

Miller 的報價前季累計為 ER=4、outs=63、ERA 約 1.71，包含後援；直接替代會得到錯誤特徵。本次 ERA 僅由逐場 `stats.pitching` 的 gamesStarted/earnedRuns/outs 計算。historical `seasonStats` 只核對登板數、先發數、ER、outs 與逐場加總一致，不作 ERA 輸入；沒有抓取或採用現在的 seasonStats。

四隊 target `gameData.teams.record` 雖出現在較早 Preview feed，卻全部符合目標比賽結束後的戰績，且比 cache 中報價前已完成場數多一場。這證明不能把 feed 的 metaData.timeStamp 一概套用到每個欄位。此 record 已排除，不用來證明完整集合。目標賽後戰績只用於此欄位洩漏核對，沒有計算方法勝率／Brier／ROI。

對 823135，截止 timecode 回傳 `20260530_052108`／Live。一次版本清單查證另列最後 `20260530_052138`；精確要求該已列版本仍回傳 `052108`／Live。兩個有限請求都沒有可用 Final，SEA 最近十場因此停止，另兩個只服務該視窗的 gamePk 未取。這不宣稱所有其他官方來源都不存在 Final。對 Montero 的 824290，要求歷史版本回傳 `20260422_013858`／Live，該 ERA 欄位停止，其他三次所需登板不再查。不能把兩個 Live 的當時比分或投球數值當 Final。

固定必要清單共 71 場歷史比賽（39 場近十場去重＋37 次投手登板，合併去重）；取得 64 場可驗證 Final、2 場 Live，5 場在相應欄位停止後未抓。33/37 次所需登板已驗證；3/4 位先發 ERA、3/4 隊十場視窗數值恢復；完整數值 1/2 場，完整 cutoff 輸入認證 **0/2**，新增前瞻樣本、方法評分及假說 **0**。

本輪共 76 次請求，74 份成功回應，raw 共 1,706,659 bytes。最初兩次 Python TLS certificate chain 失敗已保留；後續使用 macOS curl 正常 HTTPS 驗證，沒有 insecure 旗標、truststore 修改、HTTP fallback 或隱藏重試。時間清單回應成功保存後曾出現一個 print 的 AttributeError，也保留於 captures.json；其 raw SHA、收件與保存紀錄均可核對，並非網路失敗。每份 raw 都有 URL、實際 request/receipt/persist UTC 與 SHA256；2026-10-08 的本機收件不是 6 月的歷史收件，也不證明首次公告時間。

只讀離線重播（在這個 worktree 根目錄）：

```sh
python3.14 examples/two_target_feature_cutoff_v1/analyze_features.py > /tmp/match-two-target-replay.json
python3.14 examples/two_target_feature_cutoff_v1/verify_replay.py
```

重播逐份核 SHA、完整 game identity、R/2026、Final、provider version 與 play timestamps 不晚於固定 quote；再核 prior officialDate、登板角色及整季數量對帳。`verify_replay.py` 檢查 findings 逐 byte 重現、恢復欄位與既有 P52A 完全相符、Live 不納入十場數值，以及隔離副本的 raw 篡改必須拒絕。capture scripts/固定 plans 留作來源紀錄；離線重播不執行它們、不增加 GET。

`source_manifest.json` 綁定此 capsule 的固定 plans、raw、分析及查證程式。原三場 probe、主線、別人的 dirty worktree、真實 ledger/frozen predictions/正式資料庫均未修改。下一步若要解除完整集合限制，需要報價前的完整官方賽程／完成集合快照；824269 另需可信且可驗 cutoff 的缺少 Final 逐場版本。未取得前維持空值及拒絕完整輸入認證。
