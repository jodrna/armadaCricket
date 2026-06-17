import runpy
import pandas as pd
from paths import PROJECT_ROOT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle



# -------------------------
# Debug variables
# -------------------------
DEBUG_CONFIG = {
    'model': 'jungle',
    'type': 'run',
    'batsman': 'Virat Kohli',
    'host': 'India',
    'comp': 'T20I',
    'matchid': 101
}


def add_table(story, title, df, styles, rounding=2, font_size=6):
    story.append(Paragraph(title, styles['Heading2']))

    if len(df) == 0:
        story.append(Paragraph('No rows found.', styles['Normal']))
        return

    table = Table([df.columns.tolist()] + df.round(rounding).astype(str).values.tolist(), repeatRows=1)
    table.hAlign = 'LEFT'
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), ('GRID', (0, 0), (-1, -1), 0.25, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), font_size), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    story.append(table)


def get_ratings_path(debug_config):
    if debug_config['model'] == 'jungle':
        return PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsJungle3.csv'

    if debug_config['model'] == 'rasoi':
        return PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsRasoi3.csv'

    raise ValueError(f'Unknown model: {debug_config["model"]}')


def load_debug_ratings(debug_config):
    ratings = pd.read_csv(get_ratings_path(debug_config))

    ratings = ratings[
        (ratings['batsman'] == debug_config['batsman']) &
        (ratings['competition'] == debug_config['comp']) &
        (ratings['host'] == debug_config['host']) &
        (ratings['matchid'] == debug_config['matchid'])
    ].copy()

    if len(ratings) == 0:
        raise ValueError(f'No ratings row found for {debug_config}')

    return ratings.iloc[[0]]


def add_top_rating_values(story, debug_config, ratings, styles):
    ord_col = 'ord_r' if debug_config['type'] == 'run' else 'ord_w'
    cols = ['age', 'nationality', 'careerT20MatchNumber', 'balls_faced_career', ord_col]
    df = ratings.loc[:, cols].copy()
    df.columns = ['age', 'nationality', 'innings', 'balls_faced_career', 'avg_batting_position']
    df['avg_batting_position'] = df['avg_batting_position'].round(2)
    df['age'] = df['age'].round(2)
    text = ' | '.join([f'{col}: {df.iloc[0][col]}' for col in df.columns])
    story.append(Paragraph(text, styles['Normal']))


def add_reversion_rating_values(story, debug_config, ratings, styles, rounding=4, font_size=6):
    story.append(Paragraph('Player Rating Values', styles['Heading2']))

    run_cols = ['z_run_ratio', 'career_t20_run_rating', 'career_odi_run_rating', 'run_rating_0', 'run_rating', 'rep_run_ratio', 'rep_run_weight', 'run_rating_3']
    wkt_cols = ['z_wkt_ratio', 'career_t20_wkt_rating', 'career_odi_wkt_rating', 'wkt_rating_0', 'wkt_rating', 'rep_wkt_ratio', 'rep_wkt_weight', 'wkt_rating_3']
    run_values = ratings.loc[:, run_cols].round(rounding).iloc[0].tolist()
    wkt_values = ratings.loc[:, wkt_cols].round(rounding).iloc[0].tolist()

    table_data = [run_cols, run_values, wkt_cols, wkt_values]

    table = Table(table_data)
    table.hAlign = 'LEFT'
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('BACKGROUND', (0, 2), (-1, 2), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))

    story.append(table)



def make_combined_debug_pdf(debug_config, bat_model_debug, replacement_debug, ratings, output_path):
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=0.5 * cm, leftMargin=0.5 * cm, topMargin=0.5 * cm, bottomMargin=0.5 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f'{debug_config["batsman"]} Batting Report', styles['Title']))
    story.append(Paragraph(f'model: {debug_config["model"]} | type: {debug_config["type"]} | matchID: {debug_config["matchid"]} | competition: {debug_config["comp"]} | host: {debug_config["host"]}', styles['Normal']))
    add_top_rating_values(story, debug_config, ratings, styles)
    story.append(Spacer(1, 0.15 * cm))

    add_table(story, 'Rating Model: Competition / Host Breakdown', bat_model_debug['comp_summary'], styles, rounding=2, font_size=5)
    story.append(Spacer(1, 0.15 * cm))

    add_table(story, 'Rating Model: Recency Breakdown', bat_model_debug['recency_summary'], styles, rounding=2, font_size=5)
    story.append(Spacer(1, 0.15 * cm))

    story.append(Spacer(1, 0.15 * cm))

    add_table(story,f'Replacement Model: {debug_config["type"].upper()} Breakdown', replacement_debug['breakdown'], styles, rounding=4, font_size=6)
    story.append(Spacer(1, 0.15 * cm))

    if debug_config['comp'] == 'WT20I':
        add_table(story,'WT20I N2H Adjustment', replacement_debug['factor_breakdown'], styles, rounding=4, font_size=6)
        story.append(Spacer(1, 0.15 * cm))

    add_reversion_rating_values(story, debug_config, ratings, styles, rounding=4, font_size=6)

    doc.build(story)





# -------------------------
# Run bat model script
# -------------------------
print('Running bat model debug...')

bat_model_results = runpy.run_path(PROJECT_ROOT / 'men/playerRatings/batT20Mens/3_batModel.py', init_globals={'DEBUG_CONFIG': DEBUG_CONFIG})
bat_model_debug = bat_model_results['BAT_MODEL_DEBUG_TABLES']

if bat_model_debug is None:
    raise ValueError(f'No bat model debug output found for {DEBUG_CONFIG}')


# -------------------------
# Run bat replacement script
# -------------------------
print('Running bat replacement debug...')

replacement_results = runpy.run_path(PROJECT_ROOT / 'men/playerRatings/batT20Mens/4_batReplacement.py', init_globals={'DEBUG_CONFIG': DEBUG_CONFIG})
replacement_debug = replacement_results['BAT_REPLACEMENT_DEBUG_TABLES']

if replacement_debug is None:
    raise ValueError(f'No replacement debug output found for {DEBUG_CONFIG}')


# -------------------------
# Run bat reversion script
# -------------------------
print('Running bat reversion...')

runpy.run_path(PROJECT_ROOT / 'men/playerRatings/batT20Mens/5_batReversion.py')


# -------------------------
# Import ratings after reversion
# -------------------------
print('Loading ratings...')

ratings = load_debug_ratings(DEBUG_CONFIG)


# -------------------------
# Create combined PDF
# -------------------------
output_path = PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs' / f'debug_{DEBUG_CONFIG["model"]}_{DEBUG_CONFIG["type"]}_{DEBUG_CONFIG["batsman"].replace(" ", "_")}.pdf'

make_combined_debug_pdf(debug_config=DEBUG_CONFIG, bat_model_debug=bat_model_debug, replacement_debug=replacement_debug, ratings=ratings, output_path=output_path)

print(f'Saved debug PDF: {output_path}')





