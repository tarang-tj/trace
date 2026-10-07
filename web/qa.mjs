import { chromium } from '@playwright/test';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const evidence = process.env.TRACE_QA_OUTPUT || '../artifacts/studio';
await mkdir(evidence, {recursive:true});
const browser = await chromium.launch({headless:true,
  ...(process.env.TRACE_BROWSER_EXECUTABLE ? {executablePath:process.env.TRACE_BROWSER_EXECUTABLE} : {})});
const page = await browser.newPage({viewport:{width:1505,height:1045}});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => {if(message.type() === 'error') errors.push(message.text());});
const root = process.env.TRACE_URL || 'http://127.0.0.1:8765';
await page.goto(root);
await page.getByRole('button', {name:'Run inspection'}).waitFor();
await page.waitForFunction(() => document.querySelector('.image-frame img')?.complete);
await page.screenshot({path:`${evidence}/desktop-initial.png`,fullPage:false});
const initialCopy = await page.locator('body').innerText();
// innerText includes every option of a closed native select; only its selected label is visible.
const closedSelectOptions = await page.locator('select').evaluate(select =>
  [...select.options].filter(option => !option.selected).map(option => option.text));
const visibleInitialCopy = initialCopy.split('\n').map(line => line.trim())
  .filter(line => line && !closedSelectOptions.includes(line));
const allowedCopy = JSON.parse(await readFile(new URL('./allowed-copy.json', import.meta.url), 'utf8'));
assert.deepEqual(visibleInitialCopy, allowedCopy, 'Initial copy must match the ordered design specification');
assert.ok(await page.locator('footer').evaluate(footer => footer.getBoundingClientRect().bottom <= innerHeight),
  'The specified initial surface must fit the native concept viewport');
await page.getByRole('button', {name:'Run inspection'}).click();
await page.getByRole('button', {name:'Export evidence'}).waitFor({state:'visible'});
await page.waitForFunction(() => !document.querySelector('.export').disabled, {timeout:30000});
const changed = await page.locator('dd').first().innerText();
assert.match(changed,/\d+\.\d+%/);
await page.getByRole('tab', {name:'Uncertainty'}).click();
assert.equal(await page.getByRole('tab', {name:'Uncertainty'}).getAttribute('aria-selected'), 'true');
await page.getByRole('tab', {name:'Change probability'}).click();
await page.screenshot({path:`${evidence}/desktop-inspected.png`,fullPage:false});
const downloadPromise = page.waitForEvent('download');
await page.getByRole('button', {name:'Export evidence'}).click();
const download = await downloadPromise;
await download.saveAs('/tmp/trace-browser-export.json');
const exported = JSON.parse(await readFile('/tmp/trace-browser-export.json','utf8'));
assert.equal(exported.source, 'simulator');
assert.equal(exported.model_sha256.length,64);
if (process.env.TRACE_EXPECTED_SHA) assert.equal(exported.model_sha256, process.env.TRACE_EXPECTED_SHA);
assert.equal(exported.processing_size[0],96);
let reobservation = 'policy not ready';
if (exported.actions.length) {
  const firstAction = page.locator('.next-look li button').first();
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/inspect-scene') && response.request().method() === 'POST');
  await firstAction.click();
  const followup = await (await responsePromise).json();
  await page.waitForFunction(() => !document.querySelector('.export').disabled, {timeout:30000});
  assert.equal(followup.action, exported.actions[0].action);
  assert.equal(followup.reference, exported.reference);
  assert.notEqual(followup.observation, exported.observation);
  assert.equal(followup.model_sha256, exported.model_sha256);
  reobservation = 'acquired a simulated follow-up and updated inference';
}
await page.getByRole('button', {name:'New scene',exact:true}).click();
await page.waitForFunction(() => document.querySelector('.export').disabled);
await page.locator('select').selectOption('lighting');
await page.waitForFunction(() => document.querySelector('.image-frame img')?.complete);
const uploadPair = await (await page.request.post(root + '/api/scene', {data:{seed:2000088,scenario:'added'}})).json();
for(const role of ['reference','observation']) {
  await writeFile(`/tmp/trace-qa-${role}.png`, Buffer.from(uploadPair[role].split(',')[1], 'base64'));
  await page.getByLabel(role === 'reference' ? 'Reference image' : 'Observation image', {exact:true}).setInputFiles(`/tmp/trace-qa-${role}.png`);
}
await page.getByRole('button',{name:'Run inspection'}).click();
await page.waitForFunction(() => !document.querySelector('.export').disabled,{timeout:30000});
assert.ok(await page.locator('.next-look').innerText().then(text => text.includes('Simulator-trained advice')));
await page.getByRole('button',{name:'Research',exact:true}).click();
await page.getByRole('heading',{name:'Test the claim.'}).waitFor();
let research = 'page opened; frozen evaluation optional';
if (process.env.TRACE_EXPECT_RESEARCH === '1') {
  await page.getByRole('heading',{name:'Follow-up value'}).waitFor();
  await page.getByRole('heading',{name:'pcd custom test',exact:true}).waitFor();
  assert.equal(await page.locator('.experiment table').count(),7);
  assert.equal(await page.locator('.experiment table').last().locator('tbody tr').count(),7);
  assert.ok(await page.locator('.research').innerText().then(text => text.includes('Image-quality heuristic')));
  await page.screenshot({path:`${evidence}/research.png`,fullPage:true});
  research = 'six frozen detector suites and seven action strategies verified';
}
await page.getByRole('button',{name:'Inspect',exact:true}).click();
await page.getByRole('button',{name:'Local model'}).click();
assert.ok(await page.locator('.model-details').innerText().then(text => text.includes('trained checkpoint')));
await page.getByRole('button',{name:'Local model'}).click();
await page.setViewportSize({width:390,height:844});
await page.screenshot({path:`${evidence}/mobile.png`,fullPage:true});
assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
await page.setViewportSize({width:1505,height:1045});
const typography = await page.evaluate(() => Object.fromEntries(['h1','header button','.toolbar button','.evidence h2','.upload-panel input'].map(selector => {
  const style=getComputedStyle(document.querySelector(selector));
  return [selector,{size:style.fontSize,family:style.fontFamily,lineHeight:style.lineHeight,weight:style.fontWeight}];
})));
assert.deepEqual(errors,[]);
await writeFile(`${evidence}/qa.json`,JSON.stringify({viewport:[1505,1045],mobile:[390,844],errors,changed,
  reobservation,research,uploadedPair:'passed',exportedCheckpoint:exported.model_sha256,initialCopy,typography,
  aboveFoldCopyDiff:{passed:true,added:[],removed:[],renamed:[],reordered:false,closedSelectOptions}},null,2));
console.log(JSON.stringify({errors,changed,reobservation,uploadedPair:'passed',evidence},null,2));
await browser.close();
