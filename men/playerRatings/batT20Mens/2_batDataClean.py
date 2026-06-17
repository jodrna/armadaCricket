import pandas as pd
import numpy as np
from paths import PROJECT_ROOT


# -------------------------
# Imports
# -------------------------
bat_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/data/batDataCombined.csv', parse_dates=['date'])
player_info = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/playerInfo.csv', parse_dates=['dob'])
ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batRatingsFor.csv')


# -------------------------
# Optional: filter a single batter for speed
# -------------------------
# TEST_BATSMAN = 'AB de Villiers'
# bat_data = bat_data[bat_data['batsman'] == TEST_BATSMAN]
# player_info = player_info[player_info['name'] == TEST_BATSMAN]


# -------------------------
# fix some comp names
# -------------------------
bat_data['competition'] = np.where(bat_data['competition'] == 'Champions Trophy', 'ODI', bat_data['competition'])


# -------------------------
# Fix ODI player ids (mapping via cricinfo_id, then via playerid)
# -------------------------
bat_data = pd.merge(
    player_info.loc[:, ['playerid', 'cricinfo_id']],
    bat_data,
    how='right',
    left_on=['cricinfo_id'],
    right_on=['batterid'],
)

bat_data = pd.merge(
    player_info.loc[:, ['playerid', 'cricinfo_id']],
    bat_data,
    how='right',
    left_on=['playerid'],
    right_on=['batterid'],
)

bat_data['batterid'] = np.where(bat_data['playerid_x'] > 0, bat_data['playerid_x'], bat_data['playerid_y'])
bat_data = bat_data.drop(columns=['playerid_x', 'playerid_y', 'cricinfo_id_x', 'cricinfo_id_y'])


# -------------------------
# Merge batsman info (nationality, dob, batstyle etc)
# -------------------------
bat_data = pd.merge(
    player_info.loc[:, ['name', 'playerid', 'cricinfo_id', 'nationality', 'dob', 'batstyle']],
    bat_data,
    how='right',
    left_on=['playerid'],
    right_on=['batterid'],
)

bat_data = bat_data[bat_data['realexpwbat'] > 0]
bat_data = bat_data[bat_data['realexprbat'] > 0]


# -------------------------
# Dictionaries for cleaning / derived columns
# -------------------------
regions = {
    'South Africa': 'Africa',
    'Zimbabwe': 'Africa',
    'Afghanistan': 'Asia',
    'Bangladesh': 'Asia',
    'India': 'Asia',
    'Nepal': 'Asia',
    'Pakistan': 'Asia',
    'Singapore': 'Asia',
    'Sri Lanka': 'Asia',
    'United Arab Emirates': 'Asia',
    'Australia': 'AUS',
    'Germany': 'EU',
    'Italy': 'EU',
    'Netherlands': 'EU',
    'Canada': 'N America',
    'United States of America': 'N America',
    'New Zealand': 'NZ',
    'Papua New Guinea': 'PA',
    'Samoa': 'PA',
    'England': 'UK',
    'Ireland': 'UK',
    'Oman': 'Asia',
    'Scotland': 'UK',
    'Bermuda': 'WI',
    'West Indies': 'WI',
}

top_nations = {
    'Australia',
    'England',
    'India',
    'West Indies',
    'Sri Lanka',
    'Pakistan',
    'New Zealand',
    'South Africa',
    'Afghanistan',
    'Bangladesh',
}


# -------------------------
# Feature engineering / cleanup
# -------------------------
bat_data['host'] = np.where(bat_data['host'] == 'Zimbabwe (and Rhodesia)', 'Zimbabwe', bat_data['host'])
bat_data['nationality'] = np.where(bat_data['name'] == 'Tim David', 'Australia', bat_data['nationality'])

bat_data['run_ratio'] = bat_data['runs'] / bat_data['realexprbat']
bat_data['wkt_ratio'] = bat_data['wkt'] / bat_data['realexpwbat']

bat_data.insert(bat_data.columns.get_loc("ord") + 1, 'balls_faced', 1)
bat_data.insert(bat_data.columns.get_loc("host") + 1, 'host_region', bat_data['host'])
bat_data.insert(bat_data.columns.get_loc("nationality") + 1, 'home_region', bat_data['nationality'])

bat_data = bat_data.replace({'home_region': regions, 'host_region': regions}).drop(['batsman', 'batterid'], axis=1)

bat_data.insert(
    bat_data.columns.get_loc("battingteam") + 1,
    'bowlingteam',
    np.where(bat_data['home'] == bat_data['battingteam'], bat_data['away'], bat_data['home']),
)

bat_data = bat_data.rename(columns={'name': 'batsman'})

bat_data['batsman'] = np.where(bat_data['playerid'] == 527776, 'Ollie E Robinson', bat_data['batsman'])
bat_data['batsman'] = np.where(bat_data['playerid'] == 893955, 'Ollie G Robinson', bat_data['batsman'])

bat_data = bat_data[bat_data['playerid'] != 11509177]


# -------------------------
# Create dummy "today" innings across all major hosts (for current outputs)
# -------------------------
active = bat_data.loc[:, ['batsman', 'playerid', 'nationality', 'home_region', 'dob', 'batstyle']].drop_duplicates(subset=['playerid'])

active['date'] = pd.to_datetime("today")
active['date'] = active['date'].dt.normalize()

active[
    [
        'bowlerid',
        'balls_faced_innings',
        'innperiod',
        'innings',
        'ball',
        'ord',
        'balls_faced',
        'runs',
        'noball',
        'byes',
        'wkt',
        'realexprbat',
        'realexpwbat',
        'run_ratio',
        'wkt_ratio',
    ]
] = 0

active['matchid'] = 101
active['format'] = 't20'

active_hosts = pd.DataFrame(ratings[ratings['major'] == 1])
active_hosts['date'] = pd.to_datetime('today')
active_hosts['year'] = pd.to_datetime('today').year
active_hosts['date'] = active_hosts['date'].dt.normalize()

active = active.merge(active_hosts, how='left', on='date')

active['battingteam'] = np.where(active['competition'] == 'T20I', active['nationality'], np.nan)
active.loc[(active["competition"] == "T20I") & (~active["nationality"].isin(top_nations)), "competition"] = "tier_2"

bat_data = pd.concat([bat_data, active], axis=0).reset_index(drop=True)

bat_data = bat_data.sort_values(by=['playerid', 'date', 'balls_faced_innings'])
bat_data = bat_data.dropna(subset=['realexprbat', 'realexpwbat', 'playerid'], axis=0).reset_index(drop=True)


# -------------------------
# Ages + Home/Away flags
# -------------------------
bat_data['dob'] = pd.to_datetime(bat_data['dob'])

bat_data.insert(bat_data.columns.get_loc("dob") + 1, 'age', (bat_data['date'] - bat_data['dob']).dt.days / 365)
bat_data.insert(bat_data.columns.get_loc("age") + 1, 'age_round', bat_data['age'].apply(np.floor))

bat_data = bat_data[(bat_data['age'] > 0)]

bat_data.insert(bat_data.columns.get_loc("home_region") + 1, 'H/A_region', np.where(bat_data['home_region'] == bat_data['host_region'], 'Home', 'Away'))
bat_data.insert(bat_data.columns.get_loc("home_region") + 1, 'H/A_country', np.where(bat_data['nationality'] == bat_data['host'], 'Home', 'Away'))
bat_data.insert(bat_data.columns.get_loc("home_region") + 1, 'H/A_competition', np.where(bat_data['nationality'] == bat_data['host'], 'Home', 'Away'))

bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Caribbean Premier League') & (bat_data['nationality'] == 'West Indies'), 'Home', bat_data['H/A_country'])
bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Indian Premier League') & (bat_data['nationality'] == 'India'), 'Home', bat_data['H/A_competition'])
bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Pakistan Super League') & (bat_data['nationality'] == 'Pakistan'), 'Home', bat_data['H/A_competition'])
bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Afghanistan Premier League') & (bat_data['nationality'] == 'Afghanistan'), 'Home', bat_data['H/A_competition'])

bat_data['H/A_country'] = np.where((bat_data['nationality'] == 'Afghanistan') & (bat_data['host'] == 'United Arab Emirates'), 'Home', bat_data['H/A_country'])

bat_data['ipl_t20i'] = np.where((bat_data['competition'] == 'Indian Premier League') | (bat_data['competition'] == 'T20I'), 1, 0)
bat_data['home_league'] = np.where((bat_data['H/A_competition'] == 'Home') & (bat_data['competition'] != 'T20I'), 1, 0)


# -------------------------
# Collapse to innings-level for rolling sums
# -------------------------
innings_r = pd.pivot_table(
    bat_data,
    values=['balls_faced', 'ipl_t20i', 'home_league'],
    index=['playerid', 'date', 'host', 'host_region', 'H/A_country', 'competition'],
    aggfunc='sum',
).reset_index().astype({'playerid': str})

innings_r = innings_r.set_index(pd.DatetimeIndex(innings_r['date']))

career = pd.DataFrame(innings_r.groupby(['playerid'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
host = pd.DataFrame(innings_r.groupby(['playerid', 'host'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
host_region = pd.DataFrame(innings_r.groupby(['playerid', 'host_region'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
away = pd.DataFrame(innings_r.groupby(['playerid', 'H/A_country'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
competition = pd.DataFrame(innings_r.groupby(['playerid', 'competition'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
ipl_t20i = pd.DataFrame(innings_r.groupby(['playerid'])['ipl_t20i'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
home_league = pd.DataFrame(innings_r.groupby(['playerid'])['home_league'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1).replace(0, 1)

for df in [career, host, host_region, away, competition, ipl_t20i, home_league]:
    df['playerid'] = df['playerid'].astype('float')

bat_data = bat_data.merge(career.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_career'))
bat_data = bat_data.merge(host.drop_duplicates(subset=['date', 'host', 'playerid']), how='left', on=['playerid', 'host', 'date'], suffixes=('', '_host'))
bat_data = bat_data.merge(host_region.drop_duplicates(subset=['date', 'host_region', 'playerid']), how='left', on=['playerid', 'host_region', 'date'], suffixes=('', '_host_region'))
bat_data = bat_data.merge(away.drop_duplicates(subset=['date', 'H/A_country', 'playerid']), how='left', on=['playerid', 'H/A_country', 'date'], suffixes=('', '_H/A_country'))
bat_data = bat_data.merge(competition.drop_duplicates(subset=['date', 'competition', 'playerid']), how='left', on=['playerid', 'competition', 'date'], suffixes=('', '_competition'))
bat_data = bat_data.merge(ipl_t20i.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_balls_faced'))
bat_data = bat_data.merge(home_league.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_balls_faced'))


# -------------------------
# Overseas percentage adjustment vs league average
# -------------------------
bat_data['overseas_pct'] = bat_data['ipl_t20i_balls_faced'] / (bat_data['ipl_t20i_balls_faced'] + bat_data['home_league_balls_faced'])

league_overseas_pct = pd.pivot_table(
    bat_data,
    values=['overseas_pct'],
    index=['competition'],
    aggfunc='mean',
).reset_index()

bat_data = bat_data.merge(league_overseas_pct, how='left', on=['competition'])

bat_data['overseas_pct'] = bat_data['overseas_pct_x'] / bat_data['overseas_pct_y']
bat_data = bat_data.drop(labels=['overseas_pct_x', 'overseas_pct_y'], axis=1)


# -------------------------
# Career T20 match number
# -------------------------
mask = (bat_data['format'] == 't20')

bat_data['uniqueMatchMarker'] = mask & ~bat_data[mask].duplicated(subset=['playerid', 'matchid'])
bat_data['careerT20MatchNumber'] = bat_data.groupby('playerid')['uniqueMatchMarker'].cumsum()
bat_data = bat_data.drop(columns=['uniqueMatchMarker'])


# -------------------------
# Export
# -------------------------
bat_data.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/data/batDataCombinedClean.csv', index=False)
