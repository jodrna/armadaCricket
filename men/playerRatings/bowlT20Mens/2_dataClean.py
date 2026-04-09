import pandas as pd
import numpy as np
import datetime


# -------------------------
# Imports
# -------------------------
bowl_data = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/data/combinedBowlData.csv', parse_dates=['date'])   # ball by ball data
player_info = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/auxiliaries/playerInfo.csv', parse_dates=['dob'])       # date of birth, hand, nationality, bowler type etc
balls_per_match = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/data/ballsPerMatch.csv')       # average balls bowled per match for all bowlers, used in rep values
ratings = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/auxiliaries/bowlRatingsFor.csv')             # list of comps and hosts we want ratingsT20 for


# -------------------------
# Optional: filter a single bowler for speed
# -------------------------
# TEST_BOWLER = 'Sunil Narine'
# bowl_data = bowl_data[bowl_data['bowler'] == TEST_BOWLER]
# player_info = player_info[player_info['name'] == TEST_BOWLER]


# -------------------------
# Extras handling
# - Move all no-ball runs into noball column (avoid double counting)
# - Spread wide/no-ball runs across legal balls so we can drop illegal deliveries
# - Spread wickets taken on wides across legal balls so we don't lose them when dropping wides
# -------------------------
bowl_data['noball'] = np.where(bowl_data['noball'] > 0, bowl_data['runs'], bowl_data['noball'])
bowl_data['runs'] = np.where(bowl_data['noball'] > 0, 0, bowl_data['runs'])

extras_runs = pd.pivot_table(bowl_data, values=['runs', 'wide', 'noball', 'bowlerball'], index=['bowlerid', 'date', 'competition'], aggfunc='sum').reset_index()
extras_runs['extras_runs/ball'] = (extras_runs['wide'] + extras_runs['noball']) / extras_runs['bowlerball']

bowl_data = pd.merge(bowl_data, extras_runs.loc[:, ['bowlerid', 'date', 'competition', 'extras_runs/ball']], how='left', on=['bowlerid', 'date', 'competition'])
bowl_data['runs'] = bowl_data['runs'] + bowl_data['extras_runs/ball']

wide_wkt = bowl_data[(bowl_data['wide'] == 1) & (bowl_data['bowlerwicket'] == 1)].loc[:, ['bowlerid', 'date', 'competition', 'bowlerwicket']]

extras_wkt = pd.pivot_table(bowl_data, values=['bowlerball'], index=['bowlerid', 'date', 'competition'], aggfunc='sum').reset_index()
extras_wkt = extras_wkt.merge(wide_wkt, how='left', on=['bowlerid', 'date', 'competition'])
extras_wkt['extras_wkt/ball'] = extras_wkt['bowlerwicket'] / extras_wkt['bowlerball']

bowl_data = pd.merge(bowl_data, extras_wkt.loc[:, ['bowlerid', 'date', 'competition', 'extras_wkt/ball']], how='left', on=['bowlerid', 'date', 'competition'])
bowl_data['bowlerwicket'] = bowl_data['bowlerwicket'] + bowl_data['extras_wkt/ball'].fillna(0)

bowl_data = bowl_data[bowl_data['bowlerball'] > 0]
bowl_data = bowl_data[bowl_data['realexpwbowl'] > 0]
bowl_data = bowl_data[bowl_data['realexprbowl'] > 0]


# -------------------------
# Fix ODI player ids (mapping via cricinfo_id, then via playerid)
# -------------------------
bowl_data = pd.merge(player_info.loc[:, ['playerid', 'cricinfo_id']], bowl_data, how='right', left_on=['cricinfo_id'], right_on=['bowlerid'])
bowl_data = pd.merge(player_info.loc[:, ['playerid', 'cricinfo_id']], bowl_data, how='right', left_on=['playerid'], right_on=['bowlerid'])

bowl_data['bowlerid'] = np.where(bowl_data['playerid_x'] > 0, bowl_data['playerid_x'], bowl_data['playerid_y'])
bowl_data = bowl_data.drop(columns=['playerid_x', 'playerid_y', 'cricinfo_id_x', 'cricinfo_id_y'])


# -------------------------
# Merge bowler info (authoritative style comes from player_info.bowlstyle)
# -------------------------
player_info = player_info.drop_duplicates(subset=['playerid'], keep='last')

bowl_data = pd.merge(
    bowl_data,
    player_info.loc[:, ['playerid', 'name', 'cricinfo_id', 'nationality', 'dob', 'bowlstyle']],
    how='left',
    left_on=['bowlerid'],
    right_on=['playerid'],
)

bowl_data = bowl_data.rename(columns={'bowlstyle': 'bowlerstyle'})


# -------------------------
# Dictionaries for cleaning / derived columns
# -------------------------
regions = {'South Africa': 'Africa', 'Zimbabwe': 'Africa', 'Afghanistan': 'Asia', 'Bangladesh': 'Asia', 'India': 'Asia', 'Nepal': 'Asia',
           'Pakistan': 'Asia', 'Singapore': 'Asia', 'Sri Lanka': 'Asia', 'United Arab Emirates': 'Asia', 'Australia': 'AUS', 'Germany': 'EU', 'Italy': 'EU', 'Netherlands': 'EU',
           'Canada': 'N America', 'United States of America': 'N America', 'New Zealand': 'NZ', 'Papua New Guinea': 'PA', 'Samoa': 'PA', 'England': 'UK', 'Ireland': 'UK',
           'Scotland': 'UK', 'Bermuda': 'WI', 'West Indies': 'WI', 'Oman': 'Asia'}

bowler_type_1 = {'Left-arm fast': 'seam',
                'Left-arm fast-medium': 'seam',
                'Left-arm medium': 'seam',
                'Left-arm medium-fast': 'seam',
                'Left-arm slow': 'spin',
                'Legbreak': 'spin',
                'Legbreak googly': 'spin',
                'Right-arm fast': 'seam',
                'Right-arm fast-medium': 'seam',
                'Right-arm medium': 'seam',
                'Right-arm medium-fast': 'seam',
                'Right-arm offbreak': 'spin',
                'Right-arm slow': 'spin',
                'Right-arm slow-medium': 'seam',
                'Slow left-arm chinaman': 'spin',
                'Slow left-arm orthodox': 'spin',
                'Slow left-arm unorthodox': 'spin',
                'Left Orthodox': 'spin',
                'Left Pace': 'seam',
                'Left Seam': 'seam',
                'Left Unorthodox': 'spin',
                'Left-arm slow-medium': 'spin',
                'Right arm Offbreak': 'spin',
                'Right Legspin': 'spin',
                'Right Offspin': 'spin',
                'Right Pace': 'seam',
                'Right Seam': 'seam',
                'Right-arm bowler': 'seam',
                'legbreak': 'spin'}
bowler_type_2 = {'Left-arm fast': 'seam',
                'Left-arm fast-medium': 'seam',
                'Left-arm medium': 'seam',
                'Left-arm medium-fast': 'seam',
                'Left-arm slow': 'f_spin',
                'Legbreak': 'w_spin',
                'Legbreak googly': 'w_spin',
                'Right-arm fast': 'seam',
                'Right-arm fast-medium': 'seam',
                'Right-arm medium': 'seam',
                'Right-arm medium-fast': 'seam',
                'Right-arm offbreak': 'w_spin',
                'Right-arm slow': 'f_spin',
                'Right-arm slow-medium': 'seam',
                'Slow left-arm chinaman': 'w_spin',
                'Slow left-arm orthodox': 'f_spin',
                'Slow left-arm unorthodox': 'w_spin',
                'Left Orthodox': 'f_spin',
                'Left Pace': 'seam',
                'Left Seam': 'seam',
                'Left Unorthodox': 'w_spin',
                'Left-arm slow-medium': 'f_spin',
                'Right arm Offbreak': 'w_spin',
                'Right Legspin': 'w_spin',
                'Right Offspin': 'f_spin',
                'Right Pace': 'seam',
                'Right Seam': 'seam',
                'Right-arm bowler': 'seam',
                'legbreak': 'w_spin'}

bowler_arm = {'Left-arm fast': 'left_seam',
                'Left-arm fast-medium': 'left_seam',
                'Left-arm medium': 'left_seam',
                'Left-arm medium-fast': 'left_seam',
                'Left-arm slow': 'left_f_spin',
                'Right-arm fast': 'right_seam',
                'Right-arm fast-medium': 'right_seam',
                'Right-arm medium': 'right_seam',
                'Right-arm medium-fast': 'right_seam',
                'Right-arm offbreak': 'right_w_spin',
                'Right-arm slow': 'right_f_spin',
                'Right-arm slow-medium': 'right_seam',
                'Slow left-arm chinaman': 'left_w_spin',
                'Slow left-arm orthodox': 'left_f_spin',
                'Slow left-arm unorthodox': 'left_w_spin',
              'Left Orthodox': 'left_f_spin',
              'Left Pace': 'left_seam',
              'Left Seam': 'left_seam',
              'Left Unorthodox': 'left_w_spin',
              'Left-arm slow-medium': 'left_f_spin',
              'Right arm Offbreak': 'right_w_spin',
              'Right Legspin': 'right_w_spin',
              'Right Offspin': 'right_f_spin',
              'Right Pace': 'right_seam',
              'Right Seam': 'right_seam',
              'Right-arm bowler': 'right_seam'}

bowler_pace = {'Right-arm fast': 'fast',
               'Left-arm fast': 'fast'}


# -------------------------
# Feature engineering / cleanup
# -------------------------
bowl_data['host'] = np.where(bowl_data['host'] == 'Zimbabwe (and Rhodesia)', 'Zimbabwe', bowl_data['host'])

bowl_data['run_ratio'] = bowl_data['runs'] / bowl_data['realexprbowl']
bowl_data['wkt_ratio'] = bowl_data['bowlerwicket'] / bowl_data['realexpwbowl']

bowl_data.insert(bowl_data.columns.get_loc("ord") + 1, 'balls_bowled', bowl_data['bowlerball'])
bowl_data.insert(bowl_data.columns.get_loc("host") + 1, 'host_region', bowl_data['host'])
bowl_data.insert(bowl_data.columns.get_loc("nationality") + 1, 'home_region', bowl_data['nationality'])

bowl_data = bowl_data.replace({'home_region': regions, 'host_region': regions}).drop(['bowler', 'bowlerid'], axis=1)

bowl_data.insert(bowl_data.columns.get_loc("bowlerstyle") + 1, 'bowlertype_1', bowl_data['bowlerstyle'].replace(bowler_type_1))
bowl_data.insert(bowl_data.columns.get_loc("bowlerstyle") + 1, 'bowlertype_2', bowl_data['bowlerstyle'].replace(bowler_type_2))
bowl_data.insert(bowl_data.columns.get_loc("bowlertype_2") + 1, 'bowler_arm', bowl_data['bowlerstyle'].replace(bowler_arm))
bowl_data.insert(bowl_data.columns.get_loc("bowlertype_2") + 1, 'bowler_pace', bowl_data['bowlerstyle'].replace(bowler_pace))

bowl_data.insert(bowl_data.columns.get_loc("battingteam") + 1, 'bowlingteam', np.where(bowl_data['home'] == bowl_data['battingteam'], bowl_data['away'], bowl_data['home']))

bowl_data = bowl_data.rename(columns={'bowlerwicket': 'wkt', 'name': 'bowler'})

bowl_data['bowler'] = np.where(bowl_data['playerid'] == 527776, 'Ollie E Robinson', bowl_data['bowler'])
bowl_data['bowler'] = np.where(bowl_data['playerid'] == 893955, 'Ollie G Robinson', bowl_data['bowler'])


# -------------------------
# Create dummy "today" innings across all major hosts (for current outputs)
# -------------------------
active = bowl_data.loc[:, ['bowler', 'playerid', 'nationality', 'home_region', 'dob', 'bowlerstyle', 'bowlertype_1', 'bowlertype_2', 'bowler_arm', 'bowler_pace']].drop_duplicates()

active['date'] = pd.to_datetime("today")
active['date'] = active['date'].dt.normalize()

active[['batterid', 'bowlerball', 'innperiod', 'innings', 'ball', 'ord', 'balls_bowled', 'runs', 'noball', 'byes', 'wkt', 'realexprbowl', 'realexpwbowl', 'run_ratio', 'wkt_ratio']] = 0

active['matchid'] = 101
active['format'] = 't20'

active_hosts = pd.DataFrame(ratings[ratings['major'] == 1])
active_hosts['date'] = pd.to_datetime('today')
active_hosts['year'] = pd.to_datetime('today').year
active_hosts['date'] = active_hosts['date'].dt.normalize()

active = active.merge(active_hosts, how='left', on='date')

bowl_data = pd.concat([bowl_data, active], axis=0).reset_index(drop=True)

bowl_data = bowl_data.sort_values(by=['playerid', 'date', 'ball'])
bowl_data = bowl_data.dropna(subset=['realexprbowl', 'realexpwbowl', 'playerid'], axis=0).reset_index(drop=True)


# -------------------------
# Ages + Home/Away flags
# -------------------------
bowl_data['dob'] = pd.to_datetime(bowl_data['dob'])

bowl_data.insert(bowl_data.columns.get_loc("dob") + 1, 'age', (bowl_data['date'] - bowl_data['dob']).dt.days / 365)
bowl_data.insert(bowl_data.columns.get_loc("age") + 1, 'age_round', bowl_data['age'].apply(np.floor))

bowl_data = bowl_data[(bowl_data['age'] > 0)]

bowl_data.insert(bowl_data.columns.get_loc("home_region") + 1, 'H/A_region', np.where(bowl_data['home_region'] == bowl_data['host_region'], 'Home', 'Away'))
bowl_data.insert(bowl_data.columns.get_loc("home_region") + 1, 'H/A_country', np.where(bowl_data['nationality'] == bowl_data['host'], 'Home', 'Away'))
bowl_data.insert(bowl_data.columns.get_loc("home_region") + 1, 'H/A_competition', np.where(bowl_data['nationality'] == bowl_data['host'], 'Home', 'Away'))

bowl_data['H/A_competition'] = np.where((bowl_data['competition'] == 'Caribbean Premier League') & (bowl_data['nationality'] == 'West Indies'), 'Home', bowl_data['H/A_country'])
bowl_data['H/A_competition'] = np.where((bowl_data['competition'] == 'Indian Premier League') & (bowl_data['nationality'] == 'India'), 'Home', bowl_data['H/A_competition'])
bowl_data['H/A_competition'] = np.where((bowl_data['competition'] == 'Pakistan Super League') & (bowl_data['nationality'] == 'Pakistan'), 'Home', bowl_data['H/A_competition'])
bowl_data['H/A_competition'] = np.where((bowl_data['competition'] == 'Afghanistan Premier League') & (bowl_data['nationality'] == 'Afghanistan'), 'Home', bowl_data['H/A_competition'])
bowl_data['H/A_competition'] = np.where((bowl_data['competition'] == 'Vitality Blast') & (bowl_data['nationality'] == 'Scotland'), 'Home', bowl_data['H/A_competition'])

bowl_data['H/A_country'] = np.where((bowl_data['nationality'] == 'Afghanistan') & (bowl_data['host'] == 'United Arab Emirates'), 'Home', bowl_data['H/A_country'])

bowl_data['ipl_t20i'] = np.where((bowl_data['competition'] == 'Indian Premier League') | (bowl_data['competition'] == 'T20I'), 1, 0)
bowl_data['home_league'] = np.where((bowl_data['H/A_competition'] == 'Home') & (bowl_data['competition'] != 'T20I'), 1, 0)


# -------------------------
# Collapse to innings-level for rolling sums
# -------------------------
innings_r = pd.pivot_table(bowl_data, values=['balls_bowled', 'ipl_t20i', 'home_league'], index=['playerid', 'date', 'host', 'host_region', 'H/A_country', 'competition'], aggfunc='sum').reset_index().astype({'playerid': str})
innings_r = innings_r.set_index(pd.DatetimeIndex(innings_r['date']))

career = pd.DataFrame(innings_r.groupby(['playerid'])['balls_bowled'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
host = pd.DataFrame(innings_r.groupby(['playerid', 'host'])['balls_bowled'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
host_region = pd.DataFrame(innings_r.groupby(['playerid', 'host_region'])['balls_bowled'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
away = pd.DataFrame(innings_r.groupby(['playerid', 'H/A_country'])['balls_bowled'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
competition = pd.DataFrame(innings_r.groupby(['playerid', 'competition'])['balls_bowled'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
ipl_t20i = pd.DataFrame(innings_r.groupby(['playerid'])['ipl_t20i'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
home_league = pd.DataFrame(innings_r.groupby(['playerid'])['home_league'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1).replace(0, 1)

for df in [career, host, host_region, away, competition, ipl_t20i, home_league]:
    df['playerid'] = df['playerid'].astype('float')

bowl_data = bowl_data.merge(career.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_career'))
bowl_data = bowl_data.merge(host.drop_duplicates(subset=['date', 'host', 'playerid']), how='left', on=['playerid', 'host', 'date'], suffixes=('', '_host'))
bowl_data = bowl_data.merge(host_region.drop_duplicates(subset=['date', 'host_region', 'playerid']), how='left', on=['playerid', 'host_region', 'date'], suffixes=('', '_host_region'))
bowl_data = bowl_data.merge(away.drop_duplicates(subset=['date', 'H/A_country', 'playerid']), how='left', on=['playerid', 'H/A_country', 'date'], suffixes=('', '_H/A_country'))
bowl_data = bowl_data.merge(competition.drop_duplicates(subset=['date', 'competition', 'playerid']), how='left', on=['playerid', 'competition', 'date'], suffixes=('', '_competition'))
bowl_data = bowl_data.merge(ipl_t20i.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_balls_bowled'))
bowl_data = bowl_data.merge(home_league.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_balls_bowled'))


# -------------------------
# Merge balls per match + bowler level flags
# -------------------------
bowl_data = pd.merge(bowl_data, balls_per_match, how='left', on=['playerid'])

bowl_data['ballspermatch'] = bowl_data['ballspermatch'].fillna(6)

bowl_data['bowler_level'] = 'part_time'
bowl_data['bowler_level'] = np.where(bowl_data['ballspermatch'] > 10, np.where(bowl_data['ballspermatch'] > 17, 'full_time', 'mid_time'), bowl_data['bowler_level'])


# -------------------------
# Overseas percentage adjustment vs league average
# -------------------------
bowl_data['overseas_pct'] = bowl_data['ipl_t20i_balls_bowled'] / (bowl_data['ipl_t20i_balls_bowled'] + bowl_data['home_league_balls_bowled'])

league_overseas_pct = pd.pivot_table(bowl_data, values=['overseas_pct'], index=['competition'], aggfunc='mean').reset_index()

bowl_data = bowl_data.merge(league_overseas_pct, how='left', on=['competition'])

bowl_data['overseas_pct'] = bowl_data['overseas_pct_x'] / bowl_data['overseas_pct_y']
bowl_data = bowl_data.drop(labels=['overseas_pct_x', 'overseas_pct_y'], axis=1)


# -------------------------
# Bowler type sanity checks
# -------------------------
error_check = bowl_data.copy()
error_check = error_check.loc[:, ['bowler', 'bowlertype_2']]

error_check = error_check[(error_check['bowlertype_2'] != 'seam') & (error_check['bowlertype_2'] != 'f_spin') & (error_check['bowlertype_2'] != 'w_spin')]

bowler_errors = error_check['bowler'].unique()
print(bowler_errors)

types = error_check.drop_duplicates(subset=['bowler', 'bowlertype_2'])

types2 = pd.pivot_table(bowl_data, values=['date'], index=['bowlertype_2', 'bowlerstyle', 'bowler_pace', 'bowler_arm'], aggfunc='count').reset_index()


# -------------------------
# Export
# -------------------------
bowl_data.to_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/data/combinedBowlDataClean.csv', index=False)


