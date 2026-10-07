import React, {useState} from 'react';

const scenes = [['moved','Moved component'], ['removed','Removed component'], ['added','Added component'],
  ['lighting','Lighting only'], ['camera','Camera shift only'], ['blur','Blur only'], ['mixed','Mixed conditions'], ['clean','Unchanged scene']];

function ImageView({label, source}) {
  return <figure><figcaption>{label}</figcaption><div className="image-frame">{source && <img src={source} alt={`${label} view of the inspection scene`}/>}</div></figure>;
}

export default function Workspace({inspection}) {
  const [tab, setTab] = useState('probability');
  const {pair, result, busy, scenario} = inspection;
  return <div className="workspace-column">
    <section className="workspace" aria-label="Image comparison">
      <div className="toolbar">
        <label className="scene-label">Scene <select value={scenario} onChange={e => inspection.changeScenario(e.target.value)} disabled={busy}>
          {scenes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <div className="toolbar-actions"><button onClick={inspection.newScene} disabled={busy}>New scene</button>
          <button className="primary" onClick={() => inspection.inspect()} disabled={busy || !pair}>
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5 13 8 4 13.5Z" fill="none" stroke="currentColor" strokeWidth="1.2"/></svg>
            {busy ? 'Inspecting…' : 'Run inspection'}</button></div>
      </div>
      <div className="image-pair"><ImageView label="Reference" source={pair?.reference}/><ImageView label="Observation" source={pair?.observation}/></div>
      <div className="map-section">
        <div className="tabs" role="tablist" aria-label="Evidence map">
          <button role="tab" aria-selected={tab === 'probability'} onClick={() => setTab('probability')}>Change probability</button>
          <button role="tab" aria-selected={tab === 'error'} onClick={() => setTab('error')}>Uncertainty</button>
        </div>
        <div className="map-frame" role="tabpanel" aria-label={tab === 'probability' ? 'Change probability map' : 'Learned error indicator map'}>
          {(result || pair) && <img src={result ? result[tab + '_overlay'] : pair.reference} alt={result ? `${tab === 'probability' ? 'Change probability' : 'Learned error'} overlay from the trained model` : 'Reference awaiting inspection'}/>}
          {result && <div className="scale" aria-label="Map scale from zero to one"><span>1.0</span><i/><span>0.5</span><span>0.0</span></div>}
        </div>
      </div>
    </section>
    <section className="upload-panel" aria-label="Upload your pair"><h2>Upload your pair</h2><div className="upload-fields">
      {['reference','observation'].map(role => <label key={role}>{role === 'reference' ? 'Reference image' : 'Observation image'}
        <input key={`${role}-${scenario}-${pair?.seed ?? 'upload'}`} type="file" accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={e => inspection.upload(role, e.target.files[0])}/></label>)}
    </div></section>
  </div>;
}
