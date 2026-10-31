# 全量歷史價格 eligibility 與 102 場 readiness

基底 `fafbc91`。斷線前的 648 場成果已恢復並先封存於 commit `97546bd`。102 場完整分母、69／33 時間分割、38 個輸入 hash 於 commit `0af0c3a` 先封存，再執行 readiness／參考分數。沒有訓練、候選搜尋、既有 P37 分數重跑、新網路請求、監控、下注或真正 ledger／freeze 寫入。

## 價格資料的完整分母

重用原檔 `/Users/kelvin/Kelvin-WorkSpace/Betting-pool/data/tsl_odds_history.jsonl`，31,137,954 bytes，SHA256 `5604e41f817f87617956b54c4b664bbf562d496eb1c8618bd174888ef87c8efc`。沒有另複製大型原檔。官方原始 schedule SHA256 `6950ed55ff1ddeb9b90f90138847ed7ebe79ce39d37614268d344af96eb65e55`，包含 1,302 個 game records，capture metadata 的 acquisition 為 `2026-08-10T03:45:00Z`。官方日期／身分可作歷史核對，不是當時賽前已取得的 schedule。

從全量 11,400 筆出發，不依比分、模型機率或成績篩選。每筆 primary decision 保存在 `row_decisions.jsonl`。

| primary reason | 筆數 |
| --- | ---: |
| 每場最新合格 quote | 648 |
| 合格但未選中的較早／重複 quote | 6,550 |
| 非明確 MLB 隊伍名稱 | 3,264 |
| source team code 缺漏／與隊名衝突 | 508 |
| source／MLB league metadata 不足 | 82 |
| source 非 pregame | 144 |
| 官方非 Final | 168 |
| 官方同 game ID 不一致修訂 | 33 |
| 官方同隊伍／時間無配對 | 3 |

**648 個唯一 regular-season Final game**，official dates `2026-04-15`～`2026-06-22`；4/5/6 月分別 189／237／222 場。645 場 TSL start 與官方完全相同，另 3 場沿用 P28AB 一小時 crosswalk 範圍，但要求該範圍內只有一個同 home/away 隊伍官方 game；所有 ambiguity 都拒絕，不以 nearest 或 game order 強行決定。

配對結果保留 MLB provider ID、gamePk、game number、season、gameType、officialDate、start、雙邊官方 team IDs／name／code，及 raw source row index／SHA256。Source 如果另有明確 game identity，必須一致。`fetched_at` 必須嚴格早於 TSL 與官方兩個 start。

重用 P39 價格解析：只選唯一、同一 raw record、exact named sides 的雙邊 MNL。Legacy producer 把二選一與含和局的三選一都標成 MNL，三選一不混入；不編造額外 period metadata。雙邊價格必須有限且 >1；不同 source match ID／同時點價格矛盾保守排除整場。每場選最新 admissible observation，不以結果選價；若 timestamp／價格完全相同，按 raw record SHA 排序。

這是 archive 內的 local fetch timestamp，不是 provider publish time，也不是獨立外部認證。資料只有歷史價格 eligibility；不是 Champion paired、可下注價格、closing quote 或真正前瞻驗證。

## 不同於 P37 的 102 場：先固定，再評估

全量保留 official dates `2026-06-14`～`2026-06-22` 的 **102 場**。每日分母為 13／9／13／13／8／13／13／13／7。固定 development 6/14～19 **69 場**，最後 historical audit 6/20～22 **33 場**。既有公開 outcomes 與研究存在，沒有宣稱未見或 virgin holdout。Audit 只開封一次作 reference scoring；預設 CLI 不評 audit outcome。沒有新優化候選，本批 optimization trials=0。

| 純歷史參考（全量 102） | correct/n | accuracy | Brier | logloss |
| --- | --- | --- | --- | --- |
| 雙邊比例無水位市場 | 56/102 | 54.90% | 0.24447 | 0.68224 |
| 固定 p_home=0.5（tie 分類 HOME） | 57/102 | 55.88% | 0.25000 | 0.69315 |

0.5 是無資訊參考，**不是 Champion，也不是新模型或策略**。市場 development 為 42/69、Brier 0.23208、logloss 0.65633；historical audit 為 14/33、Brier 0.27036、logloss 0.73640。沒有選勝出窗口。全體市場 accuracy Wilson95 為 45.24%～64.21%；market-minus-0.5 Brier=-0.00553，9 個日期 block 的固定 seed bootstrap 描述區間 [-0.02507,+0.01765]，不支持策略改善宣稱。先前 repository 的多重嘗試與日期／隊伍相關性仍在。

## Champion 可評分分母為 0/102

固定唯一待核實模型是現有 P22B Champion；不重訓、不修改模型或 minimum edge。

| 不足證據（原因可重疊） | 場數 |
| --- | ---: |
| 完整 identity 的賽前 Champion freeze 未找到 | 102/102 |
| 原始 cached schedule 缺 starter IDs | 102/102 |
| 官方 schedule acquisition 在 target 之後 | 102/102 |
| Champion feature capture 在 cutoff 前可用未成立 | 102/102 |
| 真正 model decision time 未成立 | 102/102 |

因此 `champion_vs_market.sample_count=0`，兩邊同樣本 metrics=null；不能用以上 102 場市場參考替代 Champion 前瞻成功率。沒有把今天的官方 boxscore／probable starter lookup 回填成賽前已知特徵，因那仍無法恢復已缺失的歷史可用時間。

沿用既有 root paper_state 的參數（max quote age 120 秒、min net EV 0.02、cost/unit 0.02），完全不改策略。真正 decision freshness 為 **unknown**。僅作描述的「開賽時 quote age」為 2.95～1,432.86 分鐘，0/102 在 120 秒內；這個 proxy **不能**取代真正 freeze／decision timestamp，也不作價格 eligibility 的另一個選樣條件。沒有投注／ROI replay；CLV unknown。

## 離線重播與限制

Python 3.14、無新增依賴。從 repository root 執行；output 路徑必須尚不存在。

```sh
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.audit_tsl_archive_eligibility \
  --source /Users/kelvin/Kelvin-WorkSpace/Betting-pool/data/tsl_odds_history.jsonl \
  --expected-source-sha256 5604e41f817f87617956b54c4b664bbf562d496eb1c8618bd174888ef87c8efc \
  --schedule data/fixtures/p23f2_official_2026_history/raw/schedule_2026-03-25_2026-06-30.json \
  --expected-schedule-sha256 6950ed55ff1ddeb9b90f90138847ed7ebe79ce39d37614268d344af96eb65e55 \
  --prior-summary report/p37a_rolling_walk_forward_oos/summary.json \
  --expected-prior-sha256 f8fe76adbadbccd81ff19d7edcfe1c5ff0ac1107678db53021ba248a00249e75 \
  --output-dir /tmp/match-archive-eligibility-replay
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.audit_historical_price_readiness \
  --protocol examples/tsl_archive_eligibility_v1/readiness_protocol.json \
  --output /tmp/match-historical-readiness-replay.json --open-historical-audit
```

第一個 CLI 需要原始外部 TSL archive，bundle 不複製該 31 MB 檔；可供應相同 hash 的檔案重播。第二個 CLI 使用 repository 內的 frozen selected prices／schedule／model metadata，無外部原始 archive 需求。輸入 hash 改變就 fail；不靜默修補。完整 source/output fingerprints 在 `source_manifest.json` 與 `readiness_protocol.json`。

下一步是取得可核實的 archived pregame model/feature/decision evidence，或下一季沿用 P32A acquisition／P50C freeze／settle 記錄真正前瞻資料。目前 102 場不能用賽後新增 starter 資訊解鎖為真實 asof comparison；若改做純 retrospective reconstruction，須另立協議且保持明確標籤。沒有啟動監控或排程，50 場新例行賽 paired 計畫須等下一季而非近期承諾。
