const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html','utf8');
const slice=(a,b)=>html.slice(html.indexOf(a),html.indexOf(b,html.indexOf(a)));
function page(){const c={formatVersionBytes:n=>Math.round(n)+' B',escapeAttr:s=>String(s).replaceAll('<','&lt;')};vm.createContext(c);vm.runInContext(slice('  const I18N =','  const LANG_MAP =')+slice('  const pagingSamples =','  function renderRuntimeModels('),c);return c;}
const phase=(hits=0,misses=0)=>({phase:'decode',hits,misses,evictions:0,read_bytes:misses*100,read_us:100,lock_wait_us:0,max_width:4,microbatches:2});
const cache=(hits=0,misses=0)=>({charged_bytes:512,budget_bytes:1024,phases:[phase(hits,misses)]});
const model=(time=1000,hits=0,misses=0,instance='one',mode='ssd')=>({id:'flash',paging:{instance_id:instance,mode,sampled_unix_ms:time,experts:cache(hits,misses),ple:cache(hits,misses)}});
test('paging localizes empty, applicable, phases and I/O attribution in every language',()=>{const c=page();for(const lang of ['en','zh-Hans','zh-Hant','ja','ko']){c.m=model();c.lang=lang;const output=vm.runInContext('renderPaging(m,I18N[lang])',c);assert(!output.includes('undefined'));assert(output.includes(vm.runInContext('I18N[lang].paging_no_data',c)));assert(output.includes(vm.runInContext('I18N[lang].paging_help',c)));assert(output.includes(vm.runInContext('I18N[lang].paging_decode',c)));}});
test('paging computes deltas and resets recent samples on model reload',()=>{const c=page();c.m=model();vm.runInContext('renderPaging(m,I18N.en)',c);c.m=model(2000,3,1);let output=vm.runInContext('renderPaging(m,I18N.en)',c);assert(output.includes('75.0%'));assert(output.includes('100 B/s'));c.m=model(3000,0,0,'two');output=vm.runInContext('renderPaging(m,I18N.en)',c);assert(!output.includes('75.0%'));assert(output.includes('No data'));});
test('full expert residency still reports PLE and marks expert cache inapplicable',()=>{const c=page();c.m=model(1000,0,0,'one','full');const output=vm.runInContext('renderPaging(m,I18N.en)',c);assert(output.includes('Not applicable'));assert(output.includes('PLE row cache'));assert.equal(vm.runInContext('renderPaging({id:"other"},I18N.en)',c),'');});

test('a long polling gap cannot masquerade as a recent 30 second window',()=>{const c=page();c.m=model(1000,2,2);vm.runInContext('renderPaging(m,I18N.en)',c);c.m=model(90000,4,4);const output=vm.runInContext('renderPaging(m,I18N.en)',c);assert(output.includes('Recent hit rate (up to 30s)</span><span class="runtime-kv-detail-value">No data'));});

test('prefix persistence distinguishes pending snapshots from durable commits in every language',()=>{
  const c=page();c.el={textContent:''};c.document={getElementById:()=>c.el};
  for(const lang of ['en','zh-Hans','zh-Hant','ja','ko']){
    c.currentLang=lang;c.p={last_status:'queued',last_reason:'memory_budget',pending_jobs:1,committed_total:2,reserved_peak_bytes:1024,profile:{budget_bytes:2048},hot_memory_hits:3,pending_snapshot_hits:4,ssd_hits:5};
    vm.runInContext('renderPrefixPersistence(p)',c);
    assert(c.el.textContent.includes(vm.runInContext('I18N[currentLang].prefix_status_queued',c)));
    assert(c.el.textContent.includes('3 / 4 / 5'));
    assert(!c.el.textContent.includes('undefined'));
    c.p.last_status='committed';c.p.last_reason='';vm.runInContext('renderPrefixPersistence(p)',c);
    assert(c.el.textContent.includes(vm.runInContext('I18N[currentLang].prefix_status_committed',c)));
    vm.runInContext('renderPrefixPersistence(null)',c);assert.equal(c.el.textContent,'');
  }
});
