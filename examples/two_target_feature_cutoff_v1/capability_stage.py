from urllib.parse import urlencode
import json
from capture_source import ROOT,fetch
fields='gamePk,metaData,timeStamp,gameData,game,pk,id,type,season,datetime,dateTime,officialDate,status,abstractGameState,detailedState,teams,home,away,name,abbreviation,record,leagueRecord,wins,losses,gamesPlayed,gamesPitched,gamesStarted,winningPercentage,lastTen,last10,probablePitchers,fullName,liveData,boxscore,players,person,stats,seasonStats,pitching,earnedRuns,outs,inningsPitched,era,plays,allPlays,about,startTime,endTime'
for target in json.loads((ROOT/'plan.json').read_bytes())['targets']:
 pk=target['identity']['game_pk']
 url=f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?'+urlencode({'timecode':target['archived_timecode'],'fields':fields})
 d=fetch(url,f'raw/{pk}-feature-version.json',limit=2)
 if d:
  gd=d['gameData']; print('game',pk,'version',d['metaData'],'records',{s:v.get('record') for s,v in gd['teams'].items()},flush=True)
  for side,p in gd['probablePitchers'].items():
   player=d['liveData']['boxscore']['teams'][side]['players'].get('ID'+str(p['id']),{})
   print(side,p,'seasonStats',player.get('seasonStats',{}).get('pitching'),'currentGameStats',player.get('stats',{}).get('pitching'),flush=True)
 else: print(pk,'unavailable',flush=True)
