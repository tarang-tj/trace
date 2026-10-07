import React, {useEffect, useState} from 'react';
import {request, useInspection} from './useInspection.js';
import Workspace from './Workspace.jsx';
import Evidence from './Evidence.jsx';
import Research from './Research.jsx';

export default function App() {
  const inspection = useInspection();
  const [page, setPage] = useState('inspect'); const [health, setHealth] = useState(null); const [showModel, setShowModel] = useState(false);
  useEffect(() => {request('health').then(setHealth).catch(() => setHealth(null));}, []);
  return <><header><a className="brand" href="/" aria-label="TRACE home"><svg viewBox="0 0 28 28" width="26" height="26" aria-hidden="true"><circle cx="14" cy="14" r="6"/><path d="M14 2v8m0 8v8M2 14h8m8 0h8"/></svg><span>TRACE</span></a>
    <nav aria-label="Main navigation"><button className={page === 'inspect' ? 'selected' : ''} onClick={() => setPage('inspect')}>Inspect</button><button className={page === 'research' ? 'selected' : ''} onClick={() => setPage('research')}>Research</button></nav>
    <div className="model-control"><button aria-expanded={showModel} onClick={() => setShowModel(value => !value)}><svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d="M3 3h14v11H3ZM7 17h6m-3-3v3"/></svg>Local model<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg></button>
      {showModel && <div className="model-details">{health ? <><strong>{health.variant.toUpperCase()} · trained checkpoint</strong><p>{health.device.toUpperCase()} · 96 × 96 inference</p><code>{health.checkpoint_sha256.slice(0,16)}</code><p>Policy {health.policy_ready ? 'ready' : 'not trained'}</p></> : <p>Model unavailable. Check the local service.</p>}</div>}</div>
    </header>
    <main>{page === 'inspect' ? <><div className="intro"><h1>Look again. Know what changed.</h1><p>Compare a reference with an observation, then choose the next best look.</p></div>
      {inspection.error && <div className="error" role="alert">{inspection.error}</div>}
      <div className="inspection-layout"><Workspace inspection={inspection}/><Evidence inspection={inspection}/></div>
      <footer>Simulated scenes. Validate on captured image pairs before deployment.</footer></> : <Research/>}</main>
  </>;
}
