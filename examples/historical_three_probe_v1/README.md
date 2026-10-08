# 三場歷史來源微型查證

只做六個免費官方 GET；沒有評分、調參、批量抓取、改模型、合併或監控。目標先固定於 `2026-10-08T07:54:38.454848Z`：6/4、6/5、6/6 各取 archive 中最早 scheduled start，tie gamePk；不按結果選擇。`targets.json` SHA256 `3f691cc189a2e7bf93ec17c073a84d059c557fda5d3d36b597b8683f1e357e6d`。

每場先 GET 官方 `/api/v1.1/game/{pk}/feed/live/timestamps`，再取不晚於既有 TSL quote 時點的最後已列版本；沒有早期版本時，明確要求 quote timecode。六份 raw 共 96,017 bytes，逐份保留實際收件／保存 UTC、URL 與 SHA256。當前收件時間與 provider 歷史版本時間分開，沒有 backdate 本機紀錄。

| gamePk／日期 | 固定 quote UTC | 實際回傳官方版本 UTC | 判定 |
|---|---|---|---|
| 823457／6/4，SD @ PHI | 13:39:42.737846 | 14:06:12 | 不可恢復該 quote 時點：沒有較早已列版本，API 在要求 13:39:42 時仍回傳晚約 26 分 29 秒的版本。即使是 Preview，也不能回填。 |
| 824673／6/5，SF @ CHC | 18:18:59.881082 | 17:42:15 | 可恢復先發／部分欄位；完整 Champion inputs 不確定。Preview / Delayed Start，Robbie Ray 592662、Edward Cabrera 665795 已在較早歷史版本中可見。 |
| 824269／6/6，SEA @ DET | 15:36:02.469277 | 14:23:51 | 可恢復先發／部分欄位；完整 Champion inputs 不確定。Preview / Pre-Game，Bryce Miller 682243、Keider Montero 672456 已在較早歷史版本中可見。 |

三場均驗 gamePk/type/officialDate/gameNumber、完整隊伍 ID/name/code/start 與 archive 一致；第二、三場回傳 metaData timeStamp 在時間版本清單中，且所有回傳 plays.about 的時間早於 quote。這是官方歷史版本證據，不能聲稱取得第一個公告時間或當年 HTTP 收件紀錄，也不是前瞻樣本。

較早版本內有 `seasonStats.pitching` 的 gamesStarted／ERA／inningsPitched／earnedRuns／outs，可證明部分歷史欄位版本存在。但它不等於 P52A 的「只計先發場次」累計 ERA：沒有 prior start-only logs 或所有 appearances 都是 starts 的證據；也未取得 recent 10 completed R games 的逐場 cutoff 可用性證據。不可把整季 ERA／最新 gameLog 直接替代。因此本輪結果是 **2 場先發資訊可恢復、1 場在固定 quote 時點不可恢復；0 場完整 Champion features 已驗證**。沒有產生新方法假說或分數。

第一場停止該固定時點的驗證路線。第二、三場不能概括說只有最新版；免費官方服務確實回傳較早的賽前版本。是否能把兩項原特徵完整補齊仍不確定，本輪六 GET 預算已用盡，不擴大下載。

只讀離線重播：

```sh
python3.14 examples/historical_three_probe_v1/analyze_probe.py examples/historical_three_probe_v1 > /tmp/match-three-probe-replay.json
```

`analyze_probe.py` 核對固定 manifest／六份 raw hashes，重算逐場 identity、版本是否越過 quote 及恢復分類；輸出與 `findings.json` 逐 byte 相同。`source_manifest.json` 綁定來源與查證 scripts；原始捕捉 scripts 的六請求上限禁止在同一 capsule 追加請求。此切片是否未受其他試驗選擇影響仍未證實，不能把不在已知評分表中當作乾淨 holdout。
