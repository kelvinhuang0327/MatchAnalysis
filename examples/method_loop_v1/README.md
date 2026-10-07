# 可重複 Moneyline 方法實驗：一次候選、如實淘汰

本批基底 `a2489bf`，協議在本機 commit `e91c4de` 封存後才計算候選分數。沒有修改模型、真實 ledger、frozen predictions、正式資料庫、最低 edge 或下注規則；沒有投注、部署、push、新依賴或大批資料下載。

## 可真正重現的 baseline

先按盤口分開。原 P50C summary 的 regular-season「83 場、49 對」包含 **41 場 Moneyline（24 對）＋42 場 Run Line（25 對）**，不能拿來當 Moneyline 成功率。沿用既有 P50C market comparison 與官方日期 correction 的 read-only effective view。

| 真實 frozen Moneyline cohort | 對／final | 命中率 | Brier | logloss | 同場市場可評分 |
| --- | --- | --- | --- | --- | --- |
| 例行賽 R | 24/41 | 58.54% | 0.23702 | 0.66699 | 0/41（缺價 100%） |
| 季後賽 F | 2/4 | 50% | 0.23149 | 0.65522 | 4/4 |

例行賽命中率 Wilson 95% 區間 43.37–72.24%，不能認定已達高成功率。季後賽同 4 場市場 3 對、Brier 0.23507、logloss 0.66310；模型命中率較差而 proper scores 略好，4 場不足以認定改善。69 筆 frozen run/prediction records 中 24 筆尚無 settlement，包含重複 capture/run 的 pending records，不把它們當成新的已完成比賽。

既有 root `paper_state.json` 的規則保持不變：固定 1 單位、最低 net EV 0.02、cost 0.02/unit、quote age 120 秒。上述 4 場全為 no-bet，stake=0，ROI=null（不是 0% 獲利）；closing quote 缺少，CLV=unknown。歷史 P39 不回填這些前瞻缺價。

## 單一候選與固定時間分割

`protocol.json` 固定 8 個資料檔案 SHA256、75 場原始目標、65 場模型可評分、62 場有同場雙邊賽前 TSL MNL 報價；3/65 缺價（4.62%），全體 paired coverage 62/75（82.67%）。另 10 場因同季先發歷史不足而 model-excluded，逐場 ID／原因亦保留。無水位主勝機率為 `(1/home_odds) / (1/home_odds + 1/away_odds)`。以完整 provider/game ID/game number、官方隊伍、start、final score 驗證，不依 outcome 選價。

只試 **固定 50% incumbent＋50% market**：無訓練、無權重搜尋、無勝出窗口挑選。市場提供另一個明確可重現的訊號；是否有幫助由這次成對檢驗決定。既有 repository 已有 P37 rolling challenger 等研究，過去 trial 數非零且未知；本批新增候選數=1，不能宣稱第一次未受選擇影響的發現。

| 同樣 62 場歷史重建 MNL | 對／樣本 | 命中率 | Brier | logloss |
| --- | --- | --- | --- | --- |
| incumbent baseline | 42/62 | 67.74% | 0.22949 | 0.65157 |
| 市場 baseline | 33/62 | 53.23% | 0.24862 | 0.69068 |
| equal-pool 候選 | 40/62 | 64.52% | 0.23759 | 0.66820 |

時間順序 wf_004=development（22 paired/23 model）、wf_005=validation（15/17）、wf_006=最後 retrospective audit（25/25）。Validation 原模型 11/15、候選 10/15，Brier 0.22480→0.23347，logloss 0.64203→0.66014。Audit 皆 14/25，但 Brier 0.24510→0.24904、logloss 0.68337→0.69119。**REJECT_CANDIDATE；不改權重重試，不升版。**

既有歷史 outcome 早已出現在 repo，官方 feature 原始資料亦在目標賽後取得；這是可重現的歷史重建，不是未見或真正前瞻 holdout。預設 CLI 只評 development/validation；本批 audit 一次開封後封存，重跑僅驗證同一輸出，不能調參再選它。全體原模型命中率 Wilson95 為 55.37–78.05%。候選-minus-incumbent Brier +0.00809，按官方日期 block bootstrap 描述區間 [-0.00102,+0.01486]；僅 6 個日期 block，相關性與多重嘗試未解除，沒有顯著優化宣稱。

## 真實官方牛棚資料驗證

`official_capture/` 保存 MLB game 823218 的官方 projected feed：28,994 bytes，SHA256 `64dd93ff161800cb547817d4b3ce28270ce3e13c84e9ca15c4bed6ab226f0337`。本機收件 `2026-10-07T13:17:18.954909Z`、持久化 `2026-10-07T13:17:18.956069Z`。URL 與原始 bytes 均保留，官方資料可在離線重播；metadata 是本機收件紀錄，不是 provider 的歷史可用性認證。

UTC plays 在 6/9，球場 America/Los_Angeles 的實際單日 usage date 是 **6/8**。8 位投手逐筆與既有官方 boxscore 快取比對 outs/pitches 一致；主隊牛棚 3 outs/25 pitches，客隊 24 outs/118 pitches。使用 `stats.pitching`，不誤用 `seasonStats`；`1.2 IP=5 outs`，不以浮點乘三。先發／opener 排除，高槓桿 unknown，單場不能證明完整 3/7/14 日 coverage。

轉換沿用 legacy adapter 與 research snapshot，不另建 ledger。所有 plays 的時段作保守整場上下界，沒有冒稱個別投手精確登板時間；若多個球場日期／suspended、非 Final、缺 play timestamp、team/ID/hash 不符則拒絕。對真實歷史下一場目標的賽前 cutoff：2 位先發排除、6 位後援全部 `NOT_AVAILABLE_BEFORE_CUTOFF`，**0 個歷史可 admission**。今天抓取的資料不能回填當時賽前，也不能用這 1 場聲稱牛棚 feature 提升準確率。

## 重播（離線、無第三方依賴）

在 repository root，使用 Python 3.14；以下 output 必須是新路徑，避免覆寫任何封存結果。

```sh
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.replay_forward_moneyline_baseline --sources examples/method_loop_v1/forward_market_baseline_sources.json --output /tmp/match-forward-replay.json
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.run_moneyline_method_experiment --protocol examples/method_loop_v1/protocol.json --output /tmp/match-development-replay.json
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.run_moneyline_method_experiment --protocol examples/method_loop_v1/protocol.json --output /tmp/match-audit-replay.json --open-retrospective-audit
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.validate_official_bullpen_capture --feed examples/method_loop_v1/official_capture/game_823218.json --capture examples/method_loop_v1/official_capture/capture.json --schedule examples/method_loop_v1/official_capture/source_schedule.json --target examples/method_loop_v1/official_capture/historical_target.json --output /tmp/match-official-replay.json
```

Input hash 或 forward source membership 改變會 fail，不靜默套入修訂；新資料需另封存 baseline 版本。所有 CLI new-only output。`forward_market_baseline_sources.json` 同時綁定既有 policy、run manifests、frozen records、raw ledger、correction 及 quote artifacts；報告使用相對路徑，換 checkout 仍可 byte-identical 重播。

下一個可認定改善的 loop：沿用 P32A 真實 pregame quote acquisition／P50C freeze／既有 settle。先固定新試驗協議，再累積接下來 50 場 paired final **例行賽**，每場完整記錄缺價與 feature coverage；季後賽、不同盤口及下一季分開標示。現在只維持原模型 shadow baseline／market reference，不部署已淘汰的 equal-pool 候選。牛棚候選須有捕捉於 cutoff 前且 coverage 可用的資料才另開單一 trial；保留最後樣本，不中途挑好看的窗口。此批沒有啟動監控、下單或改策略。
