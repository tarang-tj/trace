import React from 'react';

const labels = {relight:'Relight', reregister:'Re-register', focus:'Inspect closely'};
export default function Evidence({inspection}) {
  const {result, files, busy} = inspection;
  const actions = result?.actions?.length ? result.actions : ['relight','reregister','focus'].map(action => ({action}));
  const uploaded = Boolean(files.reference || files.observation);
  const abstain = result?.actions?.length && result.actions.every(item => item.predicted_brier_reduction <= 0);
  return <aside className="evidence" aria-label="Inspection evidence">
    <h2>Evidence</h2><dl><div><dt>Changed area</dt><dd>{result ? `${(result.changed_fraction * 100).toFixed(1)}%` : '-'}</dd></div>
      <div><dt>Verification</dt><dd>{result ? `${(result.diagnostics.branch_disagreement * 100).toFixed(1)} pp` : '-'}</dd></div></dl>
    <p className="evidence-note">{result ? 'Mean disagreement across computational probes. Lower values indicate agreement.' : 'Run an inspection to view the model evidence.'}</p>
    <section className="next-look"><h2>Next best look</h2><ol>{actions.map(({action, predicted_brier_reduction}, index) => <li key={action}>
      <button disabled={!result || busy || uploaded || predicted_brier_reduction === undefined} onClick={() => inspection.inspect(action)}>
        <span className="number">{index + 1}</span><span>{labels[action]}</span>
        {predicted_brier_reduction !== undefined && <small title="Predicted reduction in Brier error">{predicted_brier_reduction >= 0 ? '+' : ''}{predicted_brier_reduction.toFixed(4)}</small>}
      </button></li>)}</ol>
      <p>{!result ? 'Recommendations appear after inspection.' : uploaded ? 'Simulator-trained advice. Capture and upload another pair to verify it.' : result.actions.length ? 'Select an action to acquire a simulated second look. Values predict Brier error reduction.' : 'Train the action policy to enable recommendations.'}</p>
      {abstain && <p>No follow-up has positive predicted value. Keep this observation, or test an action manually.</p>}
    </section>
    <button className="export" disabled={!result || busy} onClick={inspection.exportEvidence}>Export evidence</button>
  </aside>;
}
