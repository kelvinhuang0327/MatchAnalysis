# 單次真實前瞻配對

基底 `fafbc91`；本批只在獨立 worktree / branch `codex/prospective-pairing-v1` 交付。沒有合併、清理、啟動排程、真實投注、付費來源或正式 ledger 寫入。既有 102 場歷史 price-only 資料沒有重新命名為模型樣本。

## 根因與工程變更

P52A 原本將 caller 的 `as_of` 寫成 `fetched_at_utc`，未保留每個來源的真正收件／保存證據。舊入口現在明示該欄位是 inference cutoff，仍不把舊 artifact 自動補成真實收件紀錄。新 `capture_prospective_moneyline_pair` 先以 bounded transport 封存官方 schedule / starter logs，再在實際時鐘取得 cutoff；P52A 只能重播本次已保存、hash 相符的 bytes，並沿用 P50C freeze。另保存 Champion 原始 artifact、numeric features、freeze 完成時間，以及 P32A 原始來源。

比較用**實際決策時間**檢查來源可用性與報價 age，嚴格早於官方／TSL 開賽時間；不能以 `max(freeze, quote)` 推定真實決策。完整 identity 不符、同隊／時間有多個可能官方場次、缺雙邊、遲到、過期、缺 receipt 等各有明確原因。pending 不評 final；CLV 缺 closing 即 unknown。

重用既有 live quote bridge，抽出 P32A-to-frozen identity 的共同 seam；新增完整 gamePk／隊名驗證。P32A 原有 smoke 最少兩場的預設不變；新單場入口明確選擇 1 場，避免季後賽單場 slate 被工程限制擋掉。沒有改策略：最低 net EV 0.02、cost 0.02/unit、quote age 120 秒、固定 1 unit。只輸出 paper selection，沒有建立投注部位；no-bet 仍留在未來成對 accuracy / Brier / logloss 分母。

每次同步執行最多 12 個已使用的免費官方／TSL GET，45 日以內 schedule history，單回應 2 MiB / 20 秒。Mac TSL 使用既有 system curl 的正常 TLS 驗證，單次請求、無 fallback retry、無 `-k` 或憑證修改。輸出必須是 repository 外的新目錄，不覆寫既有資料。

## 實際結果與分母

2026-10-08 官方 capture 找到 849832：CLE @ CWS，D 類，官方日期 10/8，預定 UTC `2026-10-09T00:00:00Z`。這是未開賽的季後賽診斷，不是新增例行賽樣本。

`live_model_capture/` 的 4 個模型來源最後於 `07:20:29.219930Z` 保存，cutoff `2026-10-08T07:20:29.220587Z`（完整原始值以 capture_report 為準）。主隊先發 696146 本季只有 1 次先發，未滿既有 2 次；主隊 ERA / delta 保持 unknown。客隊 800048 ERA 2.5599284436，recent-win delta -0.1。**1 個官方目標、0 個 eligible Champion、1 個 model-excluded、0 個新 pending pairs、0 個新 settled pairs**。沒有補值或降低最低歷史門檻。

TSL 的正常 Python HTTPS 失敗，之後 `tsl_availability/` 用正常驗證的系統 curl 單次查證仍 exit 60。兩份失敗證據保留。這代表當前來源可用性／價格 coverage **unknown**，不是確認台彩沒有開價。0/1 個目標取得已驗證報價；模型可比較分母為 0，不能把它寫成 model eligible cohort 缺價 100%。Accuracy、Brier、logloss、ROI 都沒有新值；CLV unknown。來源 timestamp 是本機 attestation，provider publication timestamp 仍 unknown。

既有 source membership / hashes 核實後重播，與舊 `forward_market_baseline.json` 逐 byte 相同：R Moneyline 24/41，Brier 0.23702、logloss 0.66699，市場配對 0/41；F Moneyline 2/4，市場 3/4；4 場均 no-bet、stake 0、ROI null。原「83 場 49 對」是 41 Moneyline + 42 Run Line，沒有視為單一 Moneyline cohort。沒有因工程完成宣稱方法或獲利已證明。

## 重播與下一階段

只讀離線重播會驗 raw / policy / intake hashes、重算 features、Champion inference（若存在），以及 same-identity pair decisions；輸出為新檔案。此實際 model-excluded capture 可重現同一缺失特徵／排除原因。72 個聚焦測試通過；另在獨立目錄通過 12 個新測試，實際 feature replay 逐 byte 相同。12 個新合成測試包括完整單場配對、錯配、遲到、缺價、過期、no-bet、雙場歧義、資料改動與離線重播。

```sh
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.replay_prospective_moneyline_pair --capture-root examples/prospective_pairing_v1/live_model_capture --output /tmp/match-pair-replay-new.json
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.replay_forward_moneyline_baseline --sources examples/method_loop_v1/forward_market_baseline_sources.json --output /tmp/match-baseline-replay-new.json
```

有合適新場次時，先核實現有排程，再人工觸發一次以下入口；日期／gameType／gamePk 必須來自當次官方 schedule，output 不可存在。`--game-type R` 為預設；其他類型只供分開診斷。

```sh
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.capture_prospective_moneyline_pair --target-date YYYY-MM-DD --game-pk OFFICIAL_ID --output-root /tmp/unique-new-capture
```

`next_acceptance.json` 固定下一階段資料條件：唯一、同場、真實賽前的 R / MNL paired final，完整列出所有 model-excluded / 缺價 / late / stale / pending 分母，不挑成功窗口。50 只沿用 method_loop_v1 的暫定初次檢視規模，沒有正式樣本量計算依據；正式樣本量待 effect size、paired variance、日期相關及 power／precision 設計後決定，50 不能證明方法提升，也不是自動升版條件。當前先發資料不足與 TLS 信任鏈未解，且未授權監控；本批不啟動自動累積。免費來源需恢復正常驗證的 HTTPS；不關閉 TLS。等待有符合原 Champion 條件的新場次，沿用現有 settle，季後賽分開。

排程檢查只確認 exposed relevant automations = 0、相關 launch definitions = 0、舊配置 last_run 6/22 且 games 空。cron 讀取被拒未重試，未查程序清單，因此不宣稱完全沒有其他 collector。所有定義與檢查結果見 `verification.json`。

## 官方下一窗口與精確阻礙（補查）

一次 bounded official GET，查詢 `2026-10-08` 至 `2027-04-15`，`gameType=R`、只取 schedule projection；來源 47,492 bytes，SHA256 `9d6bd6d16998c56ef918870a129ac52934a88f091d447f7c74f64811d4523047`。官方截至本次 capture 的剩餘 2026 R 場次 = **0**；2027 至 4/15 列出 283 場 R，最早官方日期 **2027-03-25**。只採官方日期，不把尚未核實的 `gameDate` 時刻當成確認開賽時間。未收集未來 ERA、先發或價格。新季第一個真正 Champion eligible 日期還不知道：必須各隊有至少 10 場已完成同季 R，以及兩位已公布先發各至少 2 次同季先發。不可用 2026／季後賽補足新季門檻。

實際失敗 URL 精確為 `https://blob3rd.sportslottery.com.tw/apidata/Live/Games.zh.json`。Python `urllib.error.URLError` 包裝 `ssl.SSLCertVerificationError`：`CERTIFICATE_VERIFY_FAILED / unable to get local issuer certificate`。正常驗證的 Mac system curl 則是 `subprocess.CalledProcessError`，exit 60。兩者都在此第一個 URL 失敗，後續 Pre endpoints 未請求；不是確認 pregame 價格不存在。目前**不能確定是服務端憑證鏈不完整或本機信任鏈問題**，也不能宣稱已診斷服務端憑證錯誤。沒有關閉 TLS、修改 trust store、繞過警告或替換來源。

可執行的下一步是：在官方新季窗口、正常 HTTPS 可用且原 Champion 特徵條件成熟後，人工執行既有新入口；再沿用現有 settle 作同場成對驗證。不是現在等待 849832 的結果或開監控。`future_window/` 與 `feasibility.json` 保存官方時窗、錯誤與分母；first eligible pairing 無確切時間承諾。
