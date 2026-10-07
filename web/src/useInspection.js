import { useEffect, useRef, useState } from 'react';

export async function request(path, body) {
  const response = await fetch('/api/' + path, body instanceof FormData
    ? {method: 'POST', body} : body ? {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)} : {});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Unable to complete inspection');
  return result;
}

export function useInspection() {
  const [scenario, setScenario] = useState('moved');
  const [seed, setSeed] = useState(2000021);
  const [pair, setPair] = useState(null);
  const [result, setResult] = useState(null);
  const [files, setFiles] = useState({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const generation = useRef(0);
  const urls = useRef({});
  useEffect(() => {
    const current = ++generation.current;
    setResult(null); setError(''); setPair(null);
    request('scene', {seed, scenario}).then(value => {if(current === generation.current) setPair(value);})
      .catch(e => {if(current === generation.current) setError(e.message);});
  }, [scenario, seed]);
  useEffect(() => () => Object.values(urls.current).forEach(URL.revokeObjectURL), []);
  function upload(role, file) {
    // Invalidate pending simulator responses when switching to uploaded images.
    ++generation.current;
    if (urls.current[role]) URL.revokeObjectURL(urls.current[role]);
    const url = file ? URL.createObjectURL(file) : null;
    urls.current[role] = url;
    setFiles(previous => ({...previous, [role]: file}));
    setPair(previous => ({...previous, ...(url ? {[role]: url} : {})}));
    setResult(null); setError('');
  }
  async function inspect(action = null) {
    if (busy) return;
    const current = ++generation.current;
    setBusy(true); setError('');
    try {
      let value;
      if (files.reference || files.observation) {
        if (!files.reference || !files.observation) throw new Error('Choose both a reference and an observation image.');
        const form = new FormData();
        form.append('reference', files.reference); form.append('observation', files.observation);
        value = await request('inspect', form);
      } else value = await request('inspect-scene', {seed, scenario, action: action ?? pair?.action ?? null});
      if (current === generation.current) {setResult(value); setPair(value);}
    } catch(e) {if (current === generation.current) setError(e.message);} finally {setBusy(false);}
  }
  function clearUploads() {
    ++generation.current;
    Object.values(urls.current).forEach(URL.revokeObjectURL);
    urls.current = {};
    setFiles({});
  }
  function changeScenario(value) {clearUploads(); setScenario(value);}
  function newScene() {clearUploads(); setSeed(value => value + 1);}
  function exportEvidence() {
    const blob = new Blob([JSON.stringify(result, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url; link.download = `trace-evidence-${seed}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return {scenario, pair, result, busy, error, files, upload, inspect, changeScenario, newScene, exportEvidence};
}
