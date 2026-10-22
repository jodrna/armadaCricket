"""Detect mixed pre/post-ball runs-required alignment without repairing scores.

Only comparisons of consecutive legal-ball indices are used. The diagnostic
requires a repeated whole-match signature, not a single negative score residual.
Source inputs remain untouched. The caller may exclude flagged match IDs before
assigning chronological folds; all model comparisons must then use the new splits.
"""
import numpy as np
import pandas as pd


def audit_alignment(frame):
    """Return (match_audit, example_rows) using score transitions only.

    ``suspected_post_ball`` flags repeated evidence that runsRequired has already
    subtracted the current ball's runs while balls and wickets are pre-ball.
    ``inconsistent_transition`` identifies other negative score residuals and is
    reported separately for manual review. Neither label silently repairs data.
    """
    needed = ['matchID', 'ID', 'date', 'inningBallNumber', 'inningBallsRemaining',
              'runsRequired', 'batsmanRuns', 'totalInningWickets', 'isWicket']
    missing = [column for column in needed if column not in frame]
    if missing:
        raise ValueError('Missing score audit columns: '+', '.join(missing))
    d = frame.copy()
    if 'inningNumber' in d:
        d = d.loc[d.inningNumber.eq(2)].copy()
    d = d.sort_values(['matchID', 'inningBallNumber', 'ID']).reset_index(drop=True)
    next_row = d.groupby('matchID').shift(-1)
    consecutive = next_row.inningBallsRemaining.eq(d.inningBallsRemaining-1)
    finite = np.isfinite(d.runsRequired) & np.isfinite(next_row.runsRequired)
    integer = np.isclose(d.runsRequired, np.round(d.runsRequired), atol=1e-8, rtol=0)
    integer &= np.isclose(next_row.runsRequired, np.round(next_row.runsRequired), atol=1e-8, rtol=0)
    valid = consecutive & finite & integer
    d['preBallResidual'] = d.runsRequired-next_row.runsRequired-d.batsmanRuns
    d['postBallResidual'] = d.runsRequired-next_row.runsRequired-next_row.batsmanRuns
    d['nextRunsRequired'] = next_row.runsRequired
    d['nextBatsmanRuns'] = next_row.batsmanRuns
    d['pairEligible'] = valid.astype(int)
    d['preNegative'] = (valid & d.preBallResidual.lt(-1e-8)).astype(int)
    d['postNegative'] = (valid & d.postBallResidual.lt(-1e-8)).astype(int)
    d['preExact'] = (valid & np.isclose(d.preBallResidual, 0)).astype(int)
    d['postExact'] = (valid & np.isclose(d.postBallResidual, 0)).astype(int)
    wicket_delta = next_row.totalInningWickets-d.totalInningWickets
    d['preWicketMismatch'] = (valid & wicket_delta.ne(d.isWicket)).astype(int)
    d['postWicketMismatch'] = (valid & wicket_delta.ne(next_row.isWicket)).astype(int)
    audit = d.groupby('matchID', sort=True).agg(
        date=('date', 'min'), rows=('ID', 'size'), consecutivePairs=('pairEligible', 'sum'),
        preNegative=('preNegative', 'sum'), postNegative=('postNegative', 'sum'),
        preExact=('preExact', 'sum'), postExact=('postExact', 'sum'),
        preWicketMismatch=('preWicketMismatch', 'sum'),
        postWicketMismatch=('postWicketMismatch', 'sum'),
        firstBallsRemaining=('inningBallsRemaining', 'first'),
        firstRunsRequired=('runsRequired', 'first'), firstBatsmanRuns=('batsmanRuns', 'first'),
    ).reset_index()
    pairs = audit.consecutivePairs.replace(0, np.nan)
    audit['preNegativeRate'] = audit.preNegative/pairs
    audit['postNegativeRate'] = audit.postNegative/pairs
    audit['preExactRate'] = audit.preExact/pairs
    audit['postExactRate'] = audit.postExact/pairs
    suspected = (audit.consecutivePairs.ge(8) & audit.preNegative.ge(5)
                 & audit.preNegativeRate.ge(.10) & audit.postNegativeRate.le(.05)
                 & (audit.postExact-audit.preExact).ge(5)
                 & (audit.postExactRate-audit.preExactRate).ge(.20))
    audit['classification'] = np.select(
        [suspected, audit.preNegative.gt(0)],
        ['suspected_post_ball', 'inconsistent_transition'], default='no_negative_pre_residual')
    audit['excludeAlignment'] = suspected
    audit['exclusionReason'] = np.where(
        suspected, 'runsRequired appears post-ball while balls and wickets are pre-ball', '')
    flagged = audit.loc[suspected, 'matchID']
    # Full transitions for first two/last two flagged matches are enough to inspect
    # both historic and recent examples, without flooding the report with all rows.
    chosen = list(dict.fromkeys(flagged.head(2).tolist()+flagged.tail(2).tolist()))
    examples = d.loc[d.matchID.isin(chosen), needed+[
        'nextRunsRequired', 'nextBatsmanRuns', 'preBallResidual', 'postBallResidual',
        'pairEligible', 'preNegative', 'postNegative']].copy()
    return audit, examples


def alignment_exclusions(frame):
    """Convenience API returning the conservatively flagged match-ID set."""
    audit, _ = audit_alignment(frame)
    return set(audit.loc[audit.excludeAlignment, 'matchID'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if not args.output_directory.is_dir():
        raise ValueError('Output directory must already exist')
    audit, examples = audit_alignment(pd.read_csv(args.input))
    audit.to_csv(args.output_directory/'dataAlignmentAudit.csv', index=False)
    audit.loc[audit.excludeAlignment].to_csv(
        args.output_directory/'excludedAlignmentMatches.csv', index=False)
    examples.to_csv(args.output_directory/'dataAlignmentExamples.csv', index=False)
    print(audit.groupby('classification').agg(matches=('matchID', 'size'), rows=('rows', 'sum')).to_string())
    print('Pre-ball negative residuals:', int(audit.preNegative.sum()))
    print('Post-ball interpretation negatives among excluded:',
          int(audit.loc[audit.excludeAlignment, 'postNegative'].sum()))
