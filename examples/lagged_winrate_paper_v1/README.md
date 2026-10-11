# 保存候選的 paper 前瞻入口

模型仍為 `133d29f` 的原始 artifact（SHA256 `ce772578…7adf0`），兩模型參數與 scaler 不變，本輪 fit=0。接入現有 P50C 凍結／Final 結算及 P32A 雙邊價格配對；新增 `RESEARCH_CANDIDATE` 角色並限制隔離輸出。Champion 預設產生的 record／fingerprint 與 `133d29f` 逐項相同，不升級或替換 Champion。

實際狀態：2026-10-11 15:35 UTC，僅一次官方免費 GET 檢查 **10-11 至 10-18 的 R 賽程，未來比賽 0**；真實新增預測／Final／市場配對皆為 0。沒有持續監控或 TSL 請求。既有 schedule fixture 1,302 個表示沒有 gameInfo；完成資料僅 277 個表示、276 個 physical games（05-31 至06-22）；legacy 2,430 筆年度 schedule 也沒有 Final／完成欄位。它們不能冒充當前全季滯後輸入。

實用指令（output 必須是專案與正式 ledger 外的全新目錄）：

```sh
python3.14 examples/lagged_winrate_paper_v1/paper.py freeze --input-root /tmp/candidate-inputs --output-root /tmp/candidate-paper-GAMEPK
# 加 --capture-market 才使用既有免費 P32A/TSL 取得雙邊價格；未配價仍保留模型預測。
python3.14 examples/lagged_winrate_paper_v1/paper.py replay --output-root /tmp/candidate-paper-GAMEPK
python3.14 examples/lagged_winrate_paper_v1/paper.py settle --input-root /tmp/official-final-input --output-root /tmp/candidate-paper-GAMEPK
```

`candidate-inputs/input_manifest.json` 使用下列結構，path 均相對該目錄；placeholder 須由真實現有 captures 填入，不能手造賽前時間：

```json
{
  "schema_version": "lagged_winrate.paper_input.v1",
  "target_game_pk": "<integer>",
  "source_captures": [
    {"name":"target_schedule", "path":"target.json", "url":"<MLB official schedule URL>", "sha256":"<raw hash>", "status":"SUCCESS", "received_at_utc":"<actual receipt>", "persisted_at_utc":"<actual persist>"},
    {"name":"season_history", "path":"history.json", "url":"<full current season MLB schedule URL, hydrate=gameInfo>", "sha256":"<raw hash>", "status":"SUCCESS", "received_at_utc":"<actual receipt>", "persisted_at_utc":"<actual persist>"}
  ],
  "prior_annual": {"year":"<previous year integer>", "csv":{"path":"prior.csv","sha256":"<hash>"}, "gamelog":{"path":"prior.txt","sha256":"<hash>"}}
}
```

history URL 須為已有授權的完整 Stats API schedule、sportId=1/gameType=R，涵蓋當年01-01至目標曆日−7，不能是截短窗口或 fields 篩掉續賽資訊；provider totalGames 與原始表示數須一致。重用已有 firstPitch＋duration＋60秒保守完成上界，來源已收到 Final、沒有續賽／相衝表示，官方比賽日期及完成上界日期皆須早於7日邊界；每隊至少20場。不捏造完成UTC，未知者排除。前季沿用已完整核對的2024/2025原檔 hash；**2027 入口還缺完整2026前季來源的資格核對與登記**，目前會明確阻擋，不能拿2025充當2026。只有新R比賽、合格前季和完整當季 captures 齊備才開始凍結證據。

模型預測與市場 quote 分開保存，input fingerprint 綁定原始來源、歷史集合、年度版本及模型 hash。P50C frozen_predictions 不含目標賽果或價格；僅實際賽前保存才可進入 prospective。normalized 原始 quotes 保留，Final 比較只用通過**實際 decision 時間**的 paired quotes；風控最低EV、quote age、cost沿用現有 policy，沒有改門檻。CLV缺closing為unknown。

Final intake 的 `input_manifest.json` 只含一筆 source_captures，指向已取得的官方 schedule JSON及實際 receipt/persist/hash。gamePk、R、officialDate、gameNumber、原scheduled start、主客ID須完整唯一匹配；Live/Preview維持pending，只有Final且比分合法才呼叫既有P50C settle。結算只寫該隔離目錄的 ledger，凍結預測不變；再結算相同Final不重複計數。未配價仍可算模型分數，但市場分母／paper ROI不能充數。

保存968場的配對不確定性：兩者都對506、都錯423、候選獨對21、基準獨對18；39個discordant pairs，exact McNemar雙尾 **p=0.7492586**（2×Binomial(39,0.5)的≤18尾機率）。+3場仍可能是抽樣波動。此分析不是新holdout，也未證實穩定或勝市場優勢；測試假設discordant signs可交換且獨立，隊伍／球季相依及歷史重用限制推論。

6項入口聚焦案例通過：賽後／晚到來源、資料不足／完成未知、7日／續賽截斷、全季窗口／前季年份、來源到特徵與概率重播、Final前pending及結算冪等／frozen bytes不變。全部合成案例為HISTORICAL_REHEARSAL，未增加forward樣本。保存的合成demo可用 replay指令核對；來源時間及比賽均明示synthetic，不能算真實前瞻證據。完成後本輪停止模型變更，沒有下注、正式ledger寫入、merge/push或發布。
