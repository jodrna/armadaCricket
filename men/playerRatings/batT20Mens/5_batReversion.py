import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# -------------------------
# Report filter
# -------------------------
player_name = 'Khawaja Nafay'
competition_filter = 'T20I'
host_filter = 'Sri Lanka'
match_id = 101

# we get some extra outputs vs just using t20 data, these are for the players who played ODI before ever playing t20

for x in np.arange(0, 2, 1):
    # read data
    bat_data = pd.read_csv(
        BASE_DIR / 'bat_t20_mens' / 'all' / 'data' / 'combinedBatDataClean.csv',
        parse_dates=['date', 'dob']
    )

    # read ratingsT20
    if x == 0:
        ratings = pd.read_csv(
            BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'batRatingsJungle2.csv',
            parse_dates=['date']
        )
    else:
        ratings = pd.read_csv(
            BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'batRatingsRasoi2.csv',
            parse_dates=['date']
        )

    bat_data = bat_data[bat_data['format'] == 't20']

    # revert the rep values for tailenders
    ratings['rep_run_ratio'] = np.where(ratings['ord_r'] > 8, (((1 - ratings['rep_run_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_r'] - 8))) + ratings['rep_run_ratio'], ratings['rep_run_ratio'])
    ratings['rep_wkt_ratio'] = np.where(ratings['ord_w'] > 8, (((1 - ratings['rep_wkt_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_w'] - 8))) + ratings['rep_wkt_ratio'], ratings['rep_wkt_ratio'])

    # filter out bat data dummies and first innings of careers
    bat_data = bat_data[(bat_data['balls_faced'] > 0) & (bat_data['balls_faced_career'] > 1)]

    # merge ratingsT20 into bat data and remove balls for which there is no rating
    bat_data = bat_data.merge(
        ratings[ratings['i_balls_faced'] > 0].loc[:, ['playerid', 'date', 'balls_faced_r', 'run_rating', 'run_rating_2', 'rep_run_ratio', 'weight_balls_r', 'balls_faced_w', 'wkt_rating', 'wkt_rating_2', 'rep_wkt_ratio', 'weight_balls_w']],
        how='left', on=['playerid', 'date'])
    bat_data = bat_data[(bat_data['run_rating'] >= 0)]
    bat_data = bat_data[(bat_data['wkt_rating'] >= 0)]

    # drop nan's or we get an error in the model
    bat_data = bat_data.dropna(subset=['rep_run_ratio', 'weight_balls_r'])
    bat_data = bat_data.dropna(subset=['rep_wkt_ratio', 'weight_balls_w'])

    # old, linear, way of doing it gave bad results
    X = bat_data[['weight_balls_r']]
    X2 = bat_data[['weight_balls_w']]
    y = pd.DataFrame(bat_data['balls_faced_career'])
    model = LinearRegression()
    model.fit(X, y)
    model2 = LinearRegression()
    model2.fit(X2, y)

    # new model with more degrees, more accurate
    poly_features = PolynomialFeatures(degree=3)
    X = bat_data[['weight_balls_r']]
    X2 = bat_data[['weight_balls_w']]
    X = poly_features.fit_transform(X)
    X2 = poly_features.fit_transform(X2)
    y = bat_data.loc[:, ['balls_faced_career']].values.ravel()
    model = LinearRegression()
    model = model.fit(X, y)
    model2 = LinearRegression()
    model2 = model2.fit(X2, y)

    bat_data['balls_faced_career_exp_r'] = model.predict(X)
    ratings.dropna(subset=['weight_balls_r'], inplace=True)
    Xp = ratings[['weight_balls_r']]
    Xp = poly_features.fit_transform(Xp)
    ratings['balls_faced_career_exp_r'] = model.predict(Xp)

    bat_data['balls_faced_career_exp_w'] = model2.predict(X2)
    ratings.dropna(subset=['weight_balls_w'], inplace=True)
    X2p = ratings[['weight_balls_w']]
    X2p = poly_features.fit_transform(X2p)
    ratings['balls_faced_career_exp_w'] = model2.predict(X2p)

    test_r = bat_data.groupby(['weight_balls_r'])['balls_faced_career_exp_r'].mean().reset_index()
    test_r.sort_values(by=['weight_balls_r'], ascending=[True], inplace=True)
    test_w = bat_data.groupby(['weight_balls_w'])['balls_faced_career_exp_w'].mean().reset_index()
    test_w.sort_values(by=['weight_balls_w'], ascending=[True], inplace=True)

    bat_data['balls_for_weight_r'] = np.minimum(bat_data['balls_faced_career'], np.maximum(0, bat_data['balls_faced_career_exp_r'] + np.minimum(400, np.maximum(0, (bat_data['weight_balls_r'] * 0.857143) - 214.2857)))) # this last clause is so we get more strict application of rep values where weighted balls are low but career balls are high, the decimals were worked out by trial and error
    bat_data['balls_for_weight_w'] = np.minimum(bat_data['balls_faced_career'], np.maximum(0, bat_data['balls_faced_career_exp_w'] + np.minimum(400, np.maximum(0, (bat_data['weight_balls_w'] * 0.857143) - 214.2857))))
    ratings['balls_for_weight_r'] = np.minimum(ratings['balls_faced_career'], np.maximum(0, ratings['balls_faced_career_exp_r'] + np.minimum(400, np.maximum(0, (ratings['weight_balls_r'] * 0.857143) - 214.2857))))
    ratings['balls_for_weight_w'] = np.minimum(ratings['balls_faced_career'], np.maximum(0, ratings['balls_faced_career_exp_w'] + np.minimum(400, np.maximum(0, (ratings['weight_balls_w'] * 0.857143) - 214.2857))))

    # # this function takes the optimised parameters (which are produced in another script), the replacement value, the pre reversion rating, balls faced, and outputs the final reverted rating
    # this is just copied from the bat_revert_WH, as reversion optimiser doesn't work well for Jungle method, apart from the k in the rep_weight_r which the optimiser found a different value for, compared to the value in _WH
    if x == 0:
        def rep_weight_r(faced, rating, rep_ratio):
            # k, a, x, y = 0.002597, 0.611901, 0.000757, 0.02
            k, a, x, y = 0.001296, 0.611901, 0.000757, 0.02


            weight = np.maximum(y, np.maximum((1 - k) ** faced, a - (x * faced))) # the minimum is calculated in the optimiser as well now
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2
    else:
        def rep_weight_r(faced, rating, rep_ratio):
            k, a, x, y = 0.001296, 0.611901, 0.000757, 0.02

            weight = np.maximum(y, np.maximum((1 - k) ** faced, a - (x * faced))) # the minimum is calculated in the optimiser as well now
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2

    if x == 0:
        def rep_weight_w(faced, rating, rep_ratio):
            k, a, x, y = 0.000942, 0.8874, 0.000957, 0.02

            weight = np.maximum(y, np.maximum((1 - k) ** faced, a - (x * faced)))  # the minimum is calculated in the optimiser as well now
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2
    else:
        def rep_weight_w(faced, rating, rep_ratio):
            k, a, x, y = 0.000942, 0.8874, 0.000957, 0.02

            weight = np.maximum(y, np.maximum((1 - k) ** faced, a - (x * faced)))  # the minimum is calculated in the optimiser as well now
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2




    #now insert rating 3, the rating reverted to replacement value, insert it into bat_data and ratingsT20
    bat_data.insert(bat_data.columns.get_loc("rep_run_ratio") + 1, 'rep_run_weight', rep_weight_r(bat_data['balls_for_weight_r'], bat_data['run_rating'], bat_data['rep_run_ratio'])[0])
    bat_data.insert(bat_data.columns.get_loc("rep_run_weight") + 1, 'run_rating_3', rep_weight_r(bat_data['balls_for_weight_r'], bat_data['run_rating'], bat_data['rep_run_ratio'])[1])
    bat_data.insert(bat_data.columns.get_loc("rep_wkt_ratio") + 1, 'rep_wkt_weight', rep_weight_w(bat_data['balls_for_weight_w'], bat_data['wkt_rating'], bat_data['rep_wkt_ratio'])[0])
    bat_data.insert(bat_data.columns.get_loc("rep_wkt_weight") + 1, 'wkt_rating_3', rep_weight_w(bat_data['balls_for_weight_w'], bat_data['wkt_rating'], bat_data['rep_wkt_ratio'])[1])

    ratings.insert(ratings.columns.get_loc("rep_run_ratio") + 1, 'rep_run_weight', rep_weight_r(ratings['balls_for_weight_r'], ratings['run_rating'], ratings['rep_run_ratio'])[0])
    ratings.insert(ratings.columns.get_loc("rep_run_weight") + 1, 'run_rating_3', rep_weight_r(ratings['balls_for_weight_r'], ratings['run_rating'], ratings['rep_run_ratio'])[1])
    ratings.insert(ratings.columns.get_loc("rep_wkt_ratio") + 1, 'rep_wkt_weight', rep_weight_w(ratings['balls_for_weight_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'])[0])
    ratings.insert(ratings.columns.get_loc("rep_wkt_weight") + 1, 'wkt_rating_3', rep_weight_w(ratings['balls_for_weight_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'])[1])



    # Filter the DataFrame to keep only rows with the maximum date
    sql_upload = ratings.loc[ratings['date'] == ratings['date'].max()].copy()
    sql_upload.loc[:, 'last_match_date'] = ratings.loc[ratings['matchid'] != 101, 'date'].max()
    # this is done to make sure all names have ratingsT20, even if an id is matched to two names like Shaheen
    batter_names = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_mens/all/data/combinedBatDataClean.csv', parse_dates=['date']).loc[:, ['playerid', 'batsman']].drop_duplicates()
    sql_upload = sql_upload.merge(batter_names, how='left', left_on=['playerid'], right_on=['playerid'])
    # isolate the columns we want
    sql_upload = sql_upload.loc[:, ['last_match_date', 'batsman_y', 'playerid', 'host', 'ord_r', 'balls_faced_r', 'run_rating', 'wkt_rating', 'competition', 'rep_run_weight', 'run_rating_3', 'rep_wkt_weight', 'wkt_rating_3']]
    sql_upload.insert(sql_upload.columns.get_loc("wkt_rating") + 1, 'external_rating', 28) # nothing to do with the model, sam peek
    sql_upload.columns = ['last_match_date', 'batter', 'playerid', 'host', 'order', 'balls_faced', 'run_rating', 'wkt_rating', 'external_rating', 'competition', 'rep_run_weight', 'run_rating_2', 'rep_wkt_weight', 'wkt_rating_2']

    if x == 0:
        pdf_sql_upload = ratings.copy()

    # export the detailed ratingsT20 and table for sql upload
    if x == 0:
        ratings.to_csv(
            BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'batRatingsJungle3.csv',
            index=False
        )
        sql_upload.to_csv(
            BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'sqlUploadJungle.csv',
            index=False
        )
    else:
        ratings.to_csv(
            BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'batRatingsRasoi3.csv',
            index=False
        )
        sql_upload.to_csv(
            BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'sqlUploadRasoi.csv',
            index=False
        )

    rating_breakdown_year = pd.read_csv(
        BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'rating_breakdown_year.csv'
    )

    rating_breakdown_host_comp = pd.read_csv(
        BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'rating_breakdown_host_comp.csv'
    )

    rating_breakdown_format = pd.read_csv(
        BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / 'rating_breakdown_format.csv'
    )

summary_row = pdf_sql_upload.copy()
summary_row = summary_row[summary_row['batsman'] == player_name]
summary_row = summary_row[summary_row['competition'] == competition_filter]
summary_row = summary_row[summary_row['host'] == host_filter]
summary_row = summary_row.sort_values('date').tail(1)

pdf_path = BASE_DIR / 'bat_t20_mens' / 'all' / 'outputs' / f'{player_name}_report.pdf'

styles = getSampleStyleSheet()
story = []
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=landscape(A4),
    rightMargin=5,
    leftMargin=5,
    topMargin=5,
    bottomMargin=5
)

from reportlab.platypus import KeepInFrame

content = []

content.append(Paragraph(f'Player: {player_name}', styles['Heading2']))
content.append(Paragraph(f'Competition: {competition_filter}', styles['BodyText']))
content.append(Paragraph(f'Host: {host_filter}', styles['BodyText']))
content.append(Paragraph(f'Match ID: {match_id}', styles['BodyText']))
content.append(Spacer(1, 0.08 * inch))

def make_table_from_df(df, title):
    content.append(Paragraph(title, styles['Heading2']))
    content.append(Spacer(1, 0.05 * inch))

    df = df.copy()
    df = df.round(2)
    df = df.fillna('')

    data = [df.columns.tolist()] + df.values.tolist()

    available_width = landscape(A4)[0] - doc.leftMargin - doc.rightMargin
    col_width = available_width / len(df.columns)

    table = Table(data, colWidths=[col_width] * len(df.columns), repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 5),
        ('LEADING', (0, 0), (-1, -1), 5.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ('LEFTPADDING', (0, 0), (-1, -1), 1),
        ('RIGHTPADDING', (0, 0), (-1, -1), 1),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))

    content.append(table)
    content.append(Spacer(1, 0.08 * inch))

make_table_from_df(rating_breakdown_year, 'rating_breakdown_year')
make_table_from_df(rating_breakdown_host_comp, 'rating_breakdown_host_comp')
make_table_from_df(rating_breakdown_format, 'rating_breakdown_format')

if len(summary_row) > 0:
    replacement_value = round(summary_row['rep_run_ratio'].iloc[0], 2)
    reversion_weight = round(summary_row['rep_run_weight'].iloc[0], 4)
    rating_after_reversion = round(summary_row['run_rating_3'].iloc[0], 2)

    content.append(Paragraph('Summary', styles['Heading2']))
    content.append(Spacer(1, 0.05 * inch))
    content.append(Paragraph(f'replacement value: {replacement_value}', styles['BodyText']))
    content.append(Paragraph(f'reversion weight: {reversion_weight}', styles['BodyText']))
    content.append(Paragraph(f'final rating: {rating_after_reversion}', styles['BodyText']))

available_width = landscape(A4)[0] - doc.leftMargin - doc.rightMargin
available_height = landscape(A4)[1] - doc.topMargin - doc.bottomMargin

story.append(
    KeepInFrame(
        maxWidth=available_width,
        maxHeight=available_height,
        content=content,
        mode='shrink'
    )
)

doc.build(story)



