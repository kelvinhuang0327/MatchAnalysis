# 2026 固定參數跨年度驗證

沿用 `133d29f` 的兩模型參數／scaler 與 `b6d31f4` 的完成證據／7日滯後方法，**fit=0、沒有調參或挑窗口**。事先固定2026-01-01至12-31所有R比賽，排除已開封102個gamePks；它們只可作合格的已完成滯後歷史，不列為新評估樣本。

| 同一批1,931場 | 命中 | accuracy | Brier↓ | logloss↓ |
|---|---:|---:|---:|---:|
| 候選：前季強度＋當季勝率差 | 1,041 | 53.9099% | 0.247284 | 0.687697 |
| 保存的前季基準 | 1,038 | 53.7545% | 0.247693 | 0.688520 |
| 固定主隊／p=0.5參考 | 1,015 | 52.5634% | 0.250000 | 0.693147 |

相對基準僅多對3場（+0.15536個百分點），Brier −0.00040924、logloss −0.00082297。兩者都對1,020、都錯872、候選獨對21、基準獨對18；exact McNemar雙尾 **p=0.7492586**，增量仍可能只是抽樣波動。相對固定主隊多對26場（+1.34645個百分點），獨對63vs37，nominal exact p=0.012033；此比較不代表候選的新增項優勢已穩定，也不能代替市場。配對假設獨立／可交換的discordances，隊伍與時間相依、研究重用都限制推論。

候選仍選1,831主／100客，基準1,864主／67客。候選截距／前季／當季項平均絕對logit貢獻0.10808／0.03797／0.02090；截距主導1,741/1,931場，新增訊號對方向的影響仍小。没有因此改λ、窗口、特徵或Champion。

一次官方整季GET保存3,519,400 bytes、2,459個表示／2,430個physical gamePks；沒有逐場下載。29個重複physical games中，28個可按明確rescheduledFrom鏈採實際場次；舊排程的NULL比分不是相衝賽果，僅實際leaf供給比分，非NULL比分／隊伍／完成證據矛盾仍拒絕。1個真正續賽排除，另1個Final比分無效／平手排除；395場不足每隊20個合格歷史場次；102場按既定規則排除。**2,430−102−395−1−1=1,931**，排除舊102後覆蓋1,931/2,328=82.9467%。逐場原因與概率在CSV，没有填補缺值。

時間依據重用官方firstPitch＋gameDurationMinutes＋60秒的**保守重建上界**，不是捏造實際完賽UTC或當時HTTP收件時間。無法支持完成或身份的場次排除；當季歷史的officialDate及完成上界UTC日期都<=目標officialDate−7曆日，至少20場；驗證自身／未來賽果不進特徵，最小完成上界距目標原start/firstPitch為573,420秒。7個曆日不等於168小時。

訓練1,076列全屬2025、最晚2025-07-18；training fingerprint及train-only scaler與保存模型完全一致。2025前季年度最後完成距2026首場178日。**模型方法及參數到2026-10-11才建立，晚於這批2026比賽**，因此只是事後跨年度重建驗證，不能稱真正前瞻或當時可用模型。1,931場中793個gamePks在舊raw cache，783個已有Final label；另1,138不在那一個cache不代表未看過。legacy P84亦曾使用808個2026賽果；没有等效的這個保存模型整季評估，所以這次不是重做它的其他策略成績。

市場同場分母 **0**：既有正常P32A quote snapshot為4場F；P39A的62個歷史價格不作本輪合法賽前比較，raw價格缺完整game identity／真實賽前capture authority，已開封102場的價格也不能移植。未取得新歷史價格，未證勝市場或獲利；不算假ROI。

5項聚焦合成檢查通過，保存JSON／CSV逐byte重播一致，且新GET=0、fit=0：

```sh
python3.14 examples/cross_year_2026_fixed_v1/verify_replay.py
python3.14 examples/cross_year_2026_fixed_v1/test_evaluate.py
python3.14 examples/cross_year_2026_fixed_v1/evaluate.py > /tmp/match-cross-year-replay.json
```

若原2025 CSV／raw搬遷，給evaluate.py／verify_replay.py一個其目錄參數；不同hash拒絕。其他來源與模型均隨branch/bundle保留，不需再連網。main、Champion、133d29f模型、b6d31f4入口及正式ledger未改，未merge/push/delete或下注。

Retrosheet attribution: “The information used here was obtained free of charge from and is copyrighted by Retrosheet.” [Retrosheet](https://www.retrosheet.org/gamelogs/index.html)。
