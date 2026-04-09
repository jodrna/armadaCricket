import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

bat_data = pd.read_csv(BASE_DIR / 'data' / 'combinedBatData.csv', parse_dates=['date'])
player_info = pd.read_csv(BASE_DIR / 'auxiliaries' / 'playerInfo.csv', parse_dates=['dob'])
ratings = pd.read_csv(BASE_DIR / 'auxiliaries' / 'batRatingsFor.csv')


# # if we want to do a test we can select just one batsman to speed it up using this filter
# bat_data = bat_data[bat_data['batsman'] == 'AB de Villiers']

# odi's have different playerid's, fix this
bat_data = pd.merge(player_info.loc[:, ['playerid', 'cricinfo_id']], bat_data, how='right', left_on=['cricinfo_id'], right_on=['batterid'])
bat_data = pd.merge(player_info.loc[:, ['playerid', 'cricinfo_id']], bat_data, how='right', left_on=['playerid'], right_on=['batterid'])
bat_data['batterid'] = np.where(bat_data['playerid_x'] > 0, bat_data['playerid_x'], bat_data['playerid_y'])
bat_data = bat_data.drop(columns=['playerid_x', 'playerid_y', 'cricinfo_id_x', 'cricinfo_id_y'])

# merge batsman info, nationality etc then filter
bat_data = pd.merge(player_info.loc[:, ['name', 'playerid', 'cricinfo_id', 'nationality', 'dob', 'batstyle']], bat_data, how='right', left_on=['playerid'], right_on=['batterid'])
# take out the balls where there is no exp values
bat_data = bat_data[bat_data['realexpwbat'] > 0]
bat_data = bat_data[bat_data['realexprbat'] > 0]

# for changing countries to regions, the raw data has countries only, we need regions, these are the regions for each country
regions = {'South Africa': 'Africa', 'Zimbabwe': 'Africa', 'Afghanistan': 'Asia', 'Bangladesh': 'Asia', 'India': 'Asia', 'Nepal': 'Asia',
           'Pakistan': 'Asia', 'Singapore': 'Asia', 'Sri Lanka': 'Asia', 'United Arab Emirates': 'Asia', 'Australia': 'AUS', 'Germany': 'EU', 'Italy': 'EU', 'Netherlands': 'EU',
           'Canada': 'N America', 'United States of America': 'N America', 'New Zealand': 'NZ', 'Papua New Guinea': 'PA', 'Samoa': 'PA', 'England': 'UK', 'Ireland': 'UK', 'Oman': 'Asia',
           'Scotland': 'UK', 'Bermuda': 'WI', 'West Indies': 'WI'}

# insert and change some columns to add info
bat_data['host'] = np.where(bat_data['host'] == 'Zimbabwe (and Rhodesia)', 'Zimbabwe', bat_data['host'])
bat_data['nationality'] = np.where(bat_data['name'] == 'Tim David', 'Australia', bat_data['nationality'])
bat_data['run_ratio'], bat_data['wkt_ratio'] = bat_data['runs'] / bat_data['realexprbat'], bat_data['wkt'] / bat_data['realexpwbat']
bat_data.insert(bat_data.columns.get_loc("ord") + 1, 'balls_faced', 1)    # insert a new column beside the ord column, it's simply all 1's but we'll use it to calculate rolling balls faced
bat_data.insert(bat_data.columns.get_loc("host") + 1, 'host_region', bat_data['host'])   # insert a new column called host region column beside the host column, for now it's the same as host
bat_data.insert(bat_data.columns.get_loc("nationality") + 1, 'home_region', bat_data['nationality'])      # insert new column called home region beside nationality, same as nationality for now
# change the home and host region columns to the actual regions, also drop batsman and batterid because we have name and playerid which are the same
bat_data = bat_data.replace({'home_region': regions, 'host_region': regions}).drop(['batsman', 'batterid'], axis=1)
bat_data.insert(bat_data.columns.get_loc("battingteam") + 1, 'bowlingteam', np.where(bat_data['home'] == bat_data['battingteam'], bat_data['away'], bat_data['home']))  # get the bowling team
bat_data = bat_data.rename(columns={'name': 'batsman'}) # rename column, we drop batsman above but rename 'name' to batsman just because of location, the batsman dropped above was in the middle of the df
bat_data['batsman'] = np.where(bat_data['playerid'] == 527776, 'Ollie E Robinson', bat_data['batsman'])
bat_data['batsman'] = np.where(bat_data['playerid'] == 893955, 'Ollie G Robinson', bat_data['batsman'])
bat_data = bat_data[bat_data['playerid'] != 11509177]

# now create dummy data, an imaginary innings in every host on today's date, so we can get a rating for that player in that host
# first get a list of players
active = bat_data.loc[:, ['batsman', 'playerid', 'nationality', 'home_region', 'dob', 'batstyle']].drop_duplicates(subset=['playerid'])   # first get a list of all unique playerids and these columns
active['date'] = pd.to_datetime("today")   # set date to today
active['date'] = active['date'].dt.normalize() # this removes the time component from the date generated above, so we have only a date
active[['bowlerid', 'balls_faced_innings', 'innperiod', 'innings', 'ball', 'ord', 'balls_faced', 'runs', 'noball', 'byes', 'wkt', 'realexprbat', 'realexpwbat', 'run_ratio', 'wkt_ratio']] = 0
active['matchid'] = 101
active['format'] = 't20'
# now get a list of comps and hosts
active_hosts = pd.DataFrame(ratings[ratings['major'] == 1])    # there is comps in the table we don't want ratingsT20 for, afghan premier league for example, filter them out
active_hosts['date'], active_hosts['year'] = pd.to_datetime('today'), pd.to_datetime('today').year
active_hosts['date'] = active_hosts['date'].dt.normalize()
# mow merge the players and comps. When we do this merge every player who has 1 row is going to have x rows for all the comps we want ratingsT20 for
active = active.merge(active_hosts, how='left', on='date')
active['battingteam'] = np.where(active['competition'] == 'T20I', active['nationality'], np.nan)     # for t20i make the players batting team their nationality, will be used later in rep values
top_nations = {"Australia","England","India","West Indies","Sri Lanka", "Pakistan","New Zealand","South Africa","Afghanistan","Bangladesh"}
active.loc[(active["competition"] == "T20I") & (~active["nationality"].isin(top_nations)), "competition"] = "tier_2"



# tack these created balls onto the bottom of raw data so we can get todays ratingsT20
bat_data = pd.concat([bat_data, active], axis=0).reset_index(drop=True)
# sort the raw data and drop NA values
bat_data = bat_data.sort_values(by=['playerid', 'date', 'balls_faced_innings'])
bat_data = bat_data.dropna(subset=['realexprbat', 'realexpwbat', 'playerid'], axis=0).reset_index(drop=True)


# inserting helper columns which will be needed at various points of the ratingsT20 process
bat_data['dob'] = pd.to_datetime(bat_data['dob'])
bat_data.insert(bat_data.columns.get_loc("dob") + 1, 'age', (bat_data['date'] - bat_data['dob']).dt.days / 365)
bat_data.insert(bat_data.columns.get_loc("age") + 1, 'age_round', bat_data['age'].apply(np.floor))    # rounded down player age  for later use
bat_data = bat_data[(bat_data['age'] > 0)]    # there are a couple of obscure players with no dob so no age, drop them, or it'll create errors later
# create home/away columns baed on region, country, competition, will be important in replacement values and weightings
bat_data.insert(bat_data.columns.get_loc("home_region") + 1, 'H/A_region', np.where(bat_data['home_region'] == bat_data['host_region'], 'Home', 'Away'))
bat_data.insert(bat_data.columns.get_loc("home_region") + 1, 'H/A_country', np.where(bat_data['nationality'] == bat_data['host'], 'Home', 'Away'))
bat_data.insert(bat_data.columns.get_loc("home_region") + 1, 'H/A_competition', np.where(bat_data['nationality'] == bat_data['host'], 'Home', 'Away'))
# some comps play in different countries, IPL in UAE for example, but I still want this to be home comp for indian players
bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Caribbean Premier League') & (bat_data['nationality'] == 'West Indies'), 'Home', bat_data['H/A_country'])
bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Indian Premier League') & (bat_data['nationality'] == 'India'), 'Home', bat_data['H/A_competition'])
bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Pakistan Super League') & (bat_data['nationality'] == 'Pakistan'), 'Home', bat_data['H/A_competition'])
bat_data['H/A_competition'] = np.where((bat_data['competition'] == 'Afghanistan Premier League') & (bat_data['nationality'] == 'Afghanistan'), 'Home', bat_data['H/A_competition'])
# set UAE to be the home country for afghan players, this will be important in the host adjustment
bat_data['H/A_country'] = np.where((bat_data['nationality'] == 'Afghanistan') & (bat_data['host'] == 'United Arab Emirates'), 'Home', bat_data['H/A_country'])
# set a 1 for ipl/t20i and home league balls, we'll use both for replacement values when we look at % of balls faced in the ipl or internationals
bat_data['ipl_t20i'] = np.where((bat_data['competition'] == 'Indian Premier League') | (bat_data['competition'] == 'T20I'), 1, 0)
bat_data['home_league'] = np.where((bat_data['H/A_competition'] == 'Home') & (bat_data['competition'] != 'T20I'), 1, 0)

# just combine all balls for each batsman on the same date, effectively create innings by innings instead of ball by ball, this means less rows and faster when doing a rolling sum but the same result in the end
innings_r = pd.pivot_table(bat_data, values=['balls_faced', 'ipl_t20i', 'home_league'], index=['playerid', 'date', 'host', 'host_region', 'H/A_country', 'competition'], aggfunc='sum').reset_index().astype({'playerid': str})
innings_r = innings_r.set_index(pd.DatetimeIndex(innings_r['date']))

# rolling balls faced by career, host, region, home/away, ipl/t20i and home league at the beginning of each inning
career = pd.DataFrame(innings_r.groupby(['playerid'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
host = pd.DataFrame(innings_r.groupby(['playerid', 'host'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
host_region = pd.DataFrame(innings_r.groupby(['playerid', 'host_region'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
away = pd.DataFrame(innings_r.groupby(['playerid', 'H/A_country'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
competition = pd.DataFrame(innings_r.groupby(['playerid', 'competition'])['balls_faced'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
ipl_t20i = pd.DataFrame(innings_r.groupby(['playerid'])['ipl_t20i'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1)
home_league = pd.DataFrame(innings_r.groupby(['playerid'])['home_league'].rolling(10000, min_periods=1, closed='left').sum()).reset_index().fillna(1).replace(0, 1)

# change player id to float, it is currently a string, then merge in the balls faced by each player at each point into the ball by ball data
for x in [career, host, host_region, away, competition, ipl_t20i, home_league]:
    x['playerid'] = x['playerid'].astype('float')
bat_data = bat_data.merge(career.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_career'))
bat_data = bat_data.merge(host.drop_duplicates(subset=['date', 'host', 'playerid']), how='left', on=['playerid', 'host', 'date'], suffixes=('', '_host'))
bat_data = bat_data.merge(host_region.drop_duplicates(subset=['date', 'host_region', 'playerid']), how='left', on=['playerid', 'host_region', 'date'], suffixes=('', '_host_region'))
bat_data = bat_data.merge(away.drop_duplicates(subset=['date', 'H/A_country', 'playerid']), how='left', on=['playerid', 'H/A_country', 'date'], suffixes=('', '_H/A_country'))
bat_data = bat_data.merge(competition.drop_duplicates(subset=['date', 'competition', 'playerid']), how='left', on=['playerid', 'competition', 'date'], suffixes=('', '_competition'))
bat_data = bat_data.merge(ipl_t20i.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_balls_faced'))
bat_data = bat_data.merge(home_league.drop_duplicates(subset=['date', 'playerid']), how='left', on=['playerid', 'date'], suffixes=('', '_balls_faced'))

# work out percentage of balls in ipl and internationals for home players in domestic leagues vs the league average
bat_data['overseas_pct'] = bat_data['ipl_t20i_balls_faced'] / (bat_data['ipl_t20i_balls_faced'] + bat_data['home_league_balls_faced'])
league_overseas_pct = pd.pivot_table(bat_data, values=['overseas_pct'], index=['competition'], aggfunc='mean').reset_index()   # get the league average
#league_overseas_pct.to_csv(fr'{user_name}\OneDrive - Decimal Data Services Ltd\PythonData\Jordan\league_overseas_pct_o.csv', index=False)
bat_data = bat_data.merge(league_overseas_pct, how='left', on=['competition'])   # merge in the league average
bat_data['overseas_pct'] = bat_data['overseas_pct_x'] / bat_data['overseas_pct_y']   # now calculate player vs league average
bat_data = bat_data.drop(labels=['overseas_pct_x', 'overseas_pct_y'], axis=1)

# only consider rows where format == 't20' so we only count t20 matches
mask = (bat_data['format'] == 't20')
# Create a unique game flag: only True for the first (playerid, gameid) where format == 't20', group by player and do a cumulative sum
bat_data['uniqueMatchMarker'] = mask & ~bat_data[mask].duplicated(subset=['playerid', 'matchid'])
bat_data['careerT20MatchNumber'] = bat_data.groupby('playerid')['uniqueMatchMarker'].cumsum()
bat_data = bat_data.drop(columns=['uniqueMatchMarker'])


# export the clean bat data
bat_data.to_csv(BASE_DIR / 'data' / 'combinedBatDataClean.csv', index=False)



