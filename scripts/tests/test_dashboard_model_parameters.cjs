const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const extract = (start, end) => html.slice(html.indexOf(`  function ${start}(`),
  html.indexOf(`  function ${end}(`));

function dashboard() {
  let payload;
  const modelID = 'mlx-community/Qwen3.5-4B-4bit';
  const context = {
    closeParamsModal() {},
    window: {
      __CURRENT_PARAM_MODEL__: modelID,
      __LOCAL_MODELS__: { [modelID]: { capabilities: { supports_mtp: false } } },
      webkit: { messageHandlers: {
        saveModelParams: { postMessage(value) { payload = JSON.parse(value); } },
      } },
    },
    document: { getElementById(id) {
      return { value: id === 'modal-max-tokens' ? '4096'
        : id === 'modal-context-size' ? '262144' : '', checked: false };
    } },
    I18N: { en: {
      err_settings_persist_failed: html.match(/err_settings_persist_failed: "(IronMLX 无法保存[^"\n]+)"/)[1],
      err_model_parameters_invalid: html.match(/err_model_parameters_invalid: "(模型参数无效[^"\n]+)"/)[1],
    } },
    currentLang: 'en',
    parsePositiveInteger: value => /^\d+$/.test(value) && Number(value) > 0 ? Number(value) : null,
    showToast(message) { throw Error(message); },
    localizeBackendErrorMessage: value => value,
  };
  vm.createContext(context);
  vm.runInContext(extract('saveModelParams', 'onModelParamsSaved'), context);
  vm.runInContext(extract('localizeErrorResult', 'localizeWarningResult'), context);
  return { context, get payload() { return payload; } };
}

test('saving an ordinary model sends absent draft IDs and the selected context limit', () => {
  const page = dashboard();
  vm.runInContext('saveModelParams()', page.context);
  assert.equal(page.payload.mtp_model_id, null);
  assert.equal(page.payload.dflash2_model_id, null);
  assert.equal(page.payload.max_tokens, '4096');
});

test('parameter validation identifies the field without suggesting filesystem repair', () => {
  const { context } = dashboard();
  assert.equal(context.localizeErrorResult({ code: 'model_parameters_invalid', field: 'max_tokens' }),
    '模型参数无效：max_tokens，未保存更改。');
  assert.equal(context.localizeErrorResult({ code: 'settings_persist_failed' }),
    'IronMLX 无法保存应用设置，请检查文件权限和可用磁盘空间。');
});
