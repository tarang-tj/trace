import React, {useEffect, useState} from 'react';
import {request} from './useInspection.js';

export default function Research() {
  const [data, setData] = useState(null); const [error, setError] = useState('');
  useEffect(() => {request('research').then(setData).catch(e => setError(e.message));}, []);
  const paired = data?.['policy-report']?.test.paired_learned_minus_baseline_ci95;
  const simplerUnresolved = paired?.fixed_action?.[0] <= 0 && paired?.quality_heuristic?.[0] <= 0;
  return <section className="research"><h1>Test the claim.</h1><p className="subtitle">Physical change should survive nuisance correction. A second look should reduce error.</p>
    <p>TRACE learns to pool four computational probes: observed, relit, registered, and combined. Its action policy predicts measured re-observation value from image evidence.</p>
    <p>This is a research hypothesis. Change detection, uncertainty estimation, and active inspection have prior work. Independent field validation remains necessary.</p>
    {error && <p role="alert">{error}</p>}
    {data?.evaluation ? Object.entries(data.evaluation.suites).map(([suite, metrics]) => <div key={suite} className="experiment"><h2>{suite.replaceAll('_',' ')}</h2>
      <div className="table-scroll"><table><thead><tr><th>Model</th><th>F1</th><th>Brier</th><th>False positive rate</th></tr></thead><tbody>{Object.entries(metrics).filter(([,value]) => value.f1 !== undefined).map(([name,value]) => <tr key={name}><th>{name}</th><td>{value.f1.toFixed(3)}</td><td>{value.brier.toFixed(4)}</td><td>{(value.false_positive_rate*100).toFixed(2)}%</td></tr>)}</tbody></table></div></div>) : <p>Evaluation results appear when the frozen-checkpoint experiment finishes.</p>}
    {data?.['policy-report'] && <div className="experiment"><h2>Follow-up value</h2>
      <p>{data['policy-report'].test.scenes} held-out simulated scenes. Positive Brier reduction means lower pixel error; negative means harm.</p>
      <div className="table-scroll"><table><thead><tr><th>Strategy</th><th>Mean Brier reduction</th></tr></thead><tbody>{[
        ['Learned', 'learned_mean_brier_reduction'], ['Learned with abstention', 'abstaining_policy_mean_brier_reduction'],
        ['Uniform random (expected)', 'uniform_random_mean_brier_reduction'], ['Fixed action (validation)', 'fixed_action_mean_brier_reduction'],
        ['Image-quality heuristic', 'quality_heuristic_mean_brier_reduction'], ['Passive', 'passive_mean_brier_reduction'],
        ['Oracle among three actions', 'oracle_mean_brier_reduction']
      ].filter(([,key]) => data['policy-report'].test[key] !== undefined).map(([name,key]) => <tr key={key}><th>{name}</th><td>{data['policy-report'].test[key].toFixed(5)}</td></tr>)}</tbody></table></div>
      <p>The learned choice worsened error in {(data['policy-report'].test.harmful_action_fraction*100).toFixed(1)}% of scenes. The oracle uses ground truth and is unavailable during inspection.</p>
      {simplerUnresolved && <p>The learned policy has no demonstrated advantage over the fixed action or the image-quality heuristic in this run.</p>}
    </div>}
    <p className="research-caveat">PCD uses a custom 60/20/20 panorama split and 96-pixel crops. These results are not comparable to published benchmark rankings.</p>
    <a href="https://arxiv.org/abs/2402.01188" target="_blank" rel="noreferrer">Related work: Segment Any Change</a>
  </section>;
}
