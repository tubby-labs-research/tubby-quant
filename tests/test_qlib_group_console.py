"""Classification service/UI boundary tests; no events, matrices, or real workers."""

import http.client
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.webapp.server import start_server
from qlib_yahoo.webapp.service import ConsoleService, ServiceError


HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
ROOT = Path(__file__).resolve().parents[1]


class GroupConsoleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / "datasets" / "prices"
        self.dataset.mkdir(parents=True)
        self.manifest = {"schema_version": 1, "status": "ready", "provider": "Stooq", "symbols": ["AAA"], "sessions": 5}
        self.save_manifest()
        (self.root / "latest.json").write_text(json.dumps({"dataset": str(self.dataset)}), encoding="utf-8")
        self.overlay = self.root / "classifications" / "sic"
        self.overlay.mkdir(parents=True)
        (self.overlay / "manifest.json").write_text("{}", encoding="utf-8")
        self.preview = {"path": str(self.overlay), "groups": ["industry", "sector", "country"],
                        "manifest_sha256": "a" * 64, "events_sha256": "b" * 64,
                        "price_source": {}, "taxonomy_metadata": {"industry": "SIC4"},
                        "availability_policy": "dated", "historical_identity_limit": "research mapping",
                        "max_age_days": 450, "status": "metadata_only"}
        self.preview_call = Mock(return_value=self.preview)
        # Contract isolation: another agent owns the preview implementation. Any
        # accidental full-verification call is deliberately fatal in this double.
        for name, replacement in (("classification_preview", self.preview_call),
                                  ("classification_info", Mock(side_effect=AssertionError("event verification in UI"))),
                                  ("read_classification_overlay", Mock(side_effect=AssertionError("event loading in UI")))):
            patcher = patch("qlib_yahoo.classification." + name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.service = ConsoleService(self.root, job_factory=Mock(side_effect=AssertionError("real worker")))

    def save_manifest(self):
        (self.dataset / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_discovery_is_direct_compatible_metadata_only(self):
        bad = self.root / "classifications" / "wrong-prices"
        bad.mkdir()
        (bad / "manifest.json").write_text("{}", encoding="utf-8")
        nested = self.root / "classifications" / "container" / "nested"
        nested.mkdir(parents=True)
        (nested / "manifest.json").write_text("{}", encoding="utf-8")
        def preview(dataset, path):
            self.assertEqual(dataset, self.dataset)
            if path == bad:
                raise DatasetError("bound to a different price snapshot")
            self.assertEqual(path, self.overlay)
            return self.preview
        self.preview_call.side_effect = preview
        response = self.service.alpha_groups(str(self.dataset), str(self.overlay))
        self.assertEqual(response["status"], "metadata_only")
        self.assertEqual(response["overlays"], [self.preview])
        self.assertEqual(response["embedded_groups"], ["market"])
        self.assertEqual(set(response["groups"]), {"market", "sector", "industry", "country"})
        self.assertEqual(len(response["diagnostics"]), 1)
        self.assertIn("different price", response["diagnostics"][0]["message"])

    def test_invalid_selected_source_has_actionable_discovery_error(self):
        self.preview_call.side_effect = DatasetError("price binding changed")
        response = self.service.alpha_groups(str(self.dataset), str(self.overlay))
        self.assertIn("Trading prices", response["error"])
        self.assertIn("Embedded groups", response["error"])
        self.assertEqual(response["groups"], ["market"])

    def test_path_types_and_outside_paths_are_actionable(self):
        with self.assertRaisesRegex(ServiceError, "path string"):
            self.service.alpha_groups(str(self.dataset), {})
        response = self.service.alpha_groups(str(self.dataset), str(self.root / "outside"))
        self.assertIn("outside the data folder", response["error"])

    def test_embedded_groups_and_subindustry(self):
        self.manifest["groups"] = {"file": "universe/groups.csv"}
        self.save_manifest()
        response = self.service.alpha_groups(str(self.dataset), "")
        self.assertEqual(set(response["groups"]), {"market", "sector", "industry", "country"})
        self.assertNotIn("subindustry", response["groups"])
        with self.assertRaisesRegex(ServiceError, "Subindustry.*unsupported"):
            self.service.start_alpha({"expressions": "rank(close)", "neutralization": "SUBINDUSTRY"})

    def test_old_recovery_options_still_fail(self):
        for key in ("classification", "classifications", "classification_path", "classification_overlay", "fast_delay", "fast_settings", "alpha_dialect", "price_dataset", "alpha_settings", "execution_phase", "deal_price", "reuse_run", "source_run"):
            with self.subTest(key=key), self.assertRaisesRegex(ServiceError, "Unsupported PR45"):
                self.service.start_alpha({"expressions": "rank(close)", key: "", "group_overlay": str(self.overlay)})
        self.preview_call.assert_not_called()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional")
    def test_all_group_operators_and_options_use_preview_then_forward(self):
        with patch.object(self.service, "_start", return_value={"ok": True}) as start, \
                patch("qlib_yahoo.dataset_integrity.verify_dataset", side_effect=AssertionError("full verify")), \
                patch("qlib_yahoo.reference.dataset_groups", side_effect=AssertionError("group CSV")):
            for group in ("sector", "industry", "country"):
                for expression in (f"group_rank(close, {group})", f"g = {group}; group_rank(close, g)"):
                    self.assertTrue(self.service.alpha_check(expression, str(self.dataset), group_overlay=str(self.overlay))["ok"])
                self.service.start_alpha({"expressions": "rank(close)", "neutralization": group.upper(), "group_overlay": str(self.overlay)})
                request = start.call_args.args[2]
                self.assertEqual(request["group_overlay"], str(self.overlay))
                self.assertEqual(request["settings"]["neutralization"], group.upper())
            self.service.start_alpha({"expressions": "group_rank(close, industry)", "group_overlay": str(self.overlay)})
            self.assertEqual(start.call_args.args[2]["group_overlay"], str(self.overlay))

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional")
    def test_irrelevant_stale_source_does_not_check_or_penalize_price_only(self):
        self.preview_call.side_effect = DatasetError("stale binding")
        with patch.object(self.service, "_start", return_value={"ok": True}) as start:
            for neutralization in ("MARKET", "NONE"):
                for expression in ("rank(close)", "group_rank(close, market)", "industry = close; rank(industry)"):
                    self.assertTrue(self.service.alpha_check(expression, str(self.dataset), group_overlay=str(self.overlay))["ok"])
                    self.service.start_alpha({"expressions": expression, "neutralization": neutralization, "group_overlay": str(self.overlay)})
                    self.assertEqual(start.call_args.args[2]["group_overlay"], str(self.overlay))
        self.preview_call.assert_not_called()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional")
    def test_invalid_source_blocks_only_group_consumers(self):
        self.preview_call.side_effect = DatasetError("different price snapshot")
        with patch.object(self.service, "_start") as start:
            checked = self.service.alpha_check("rank(close)\ngroup_rank(close, sector)", str(self.dataset), group_overlay=str(self.overlay))
            self.assertTrue(checked["lines"][0]["ok"])
            self.assertFalse(checked["lines"][1]["ok"])
            self.assertIn("Classification source unavailable", checked["lines"][1]["message"])
            for params in ({"expressions": "group_rank(close, country)"}, {"expressions": "rank(close)", "neutralization": "INDUSTRY"}):
                with self.assertRaisesRegex(ServiceError, "Trading prices"):
                    self.service.start_alpha({**params, "group_overlay": str(self.overlay)})
            start.assert_not_called()

    @unittest.skipUnless(HAVE_NUMPY, "numpy is optional")
    def test_missing_group_and_subindustry_are_not_fabricated(self):
        self.preview["groups"] = ["industry"]
        self.assertFalse(self.service.alpha_check("group_rank(close, country)", str(self.dataset), group_overlay=str(self.overlay))["ok"])
        checked = self.service.alpha_check("group_rank(close, subindustry)", str(self.dataset), group_overlay=str(self.overlay))
        self.assertIn("subindustry is unsupported", checked["lines"][0]["message"])
        with self.assertRaisesRegex(ServiceError, "COUNTRY.*needs groups"):
            self.service.start_alpha({"expressions": "rank(close)", "neutralization": "COUNTRY", "group_overlay": str(self.overlay)})

    def test_http_get_post_discovery_and_check_contract(self):
        server, _ = start_server(self.service)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        self.addCleanup(connection.close)
        headers = {"Cookie": f"tqc={server.token}", "X-Token": server.token, "Origin": server.origin, "Content-Type": "application/json"}
        from urllib.parse import urlencode
        for method in ("GET", "POST"):
            params = {"dataset": str(self.dataset), "group_overlay": str(self.overlay)}
            url = "/api/alpha-groups" + ("?" + urlencode(params) if method == "GET" else "")
            connection.request(method, url, json.dumps(params) if method == "POST" else None, headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["selected"], self.preview)
        with patch.object(self.service, "alpha_check", return_value={"ok": True, "lines": []}) as check:
            connection.request("POST", "/api/alpha-check", json.dumps({"text": "rank(close)", **params}), headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            check.assert_called_once_with("rank(close)", str(self.dataset), group_overlay=str(self.overlay))
        connection.request("POST", "/api/alpha-check", json.dumps({"text": "rank(close)", "classification_path": ""}), headers)
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        self.assertIn("Unsupported PR45", response.read().decode())


@unittest.skipUnless(shutil.which("node"), "node is optional")
class GroupUITests(unittest.TestCase):
    def test_neutralization_markup_has_stable_explicit_values(self):
        html = (ROOT / "qlib_yahoo/webapp/static/index.html").read_text(encoding="utf-8")
        select = re.search(r'<select id="lab-neutralization">(.*?)</select>', html, re.S)
        self.assertIsNotNone(select)
        options = re.findall(r'<option value="([^"]+)">([^<]+)</option>', select.group(1))
        self.assertEqual(options, [(value, value) for value in ("MARKET", "SECTOR", "INDUSTRY", "COUNTRY", "NONE")])
        self.assertEqual(select.group(1).count("<option"), len(options))

    def test_saved_selection_stale_requests_and_unsupported_neutralization(self):
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const source = fs.readFileSync('qlib_yahoo/webapp/static/app.js', 'utf8');
const controls = new Map();
function control(id) {
  if (!controls.has(id)) controls.set(id, {value:'', options:[], classList:{toggle(){}}, append(o){this.options.push(o)}, textContent:''});
  return controls.get(id);
}
const pending=[];
const context = { console, Object, JSON, Set, Map, setTimeout, clearTimeout,
  $:control, $$:()=>[], el:(tag, attrs, ...children)=>({...attrs, value:attrs.value ?? attrs.text, children:children.filter(Boolean)}),
  setChildren:(target,...children)=>{target.options=children},
  get:()=>new Promise((resolve,reject)=>pending.push({resolve,reject})),
  persist:()=>{}, onLabInput:()=>{},
};
vm.createContext(context);
const start=source.indexOf('function groupDescription'), end=source.indexOf('function setLabDelay',start);
vm.runInContext('const state={lab:{datasetPath:"prices",groupOverlay:"",groupExplicit:false,groupEpoch:0,groupIdentity:null,groupRestoreError:""}};'+source.slice(start,end), context);
// Model HTML's implicit option value and the select setter clearing selection
// when no option matches. Plain objects would miss the startup regression.
control('#lab-neutralization').options=['MARKET','NONE','SECTOR','INDUSTRY','COUNTRY'].map(text=>{
  let explicit;
  return {textContent:text, get value(){return explicit ?? this.textContent}, set value(value){explicit=String(value)}};
});
let selectedOption=null;
Object.defineProperty(control('#lab-neutralization'),'value',{
  get(){return selectedOption?.value ?? ''},
  set(value){selectedOption=this.options.find(option=>option.value===value) ?? null}
});
control('#lab-neutralization').value='INDUSTRY';
const run=code=>vm.runInContext(code,context);
assert.match(run('groupDescription({taxonomy_metadata:{industry:"Custom industry code"}}).caption'),/Custom industry code/);
assert.doesNotMatch(run('groupDescription({taxonomy_metadata:{industry:"Custom industry code"}}).caption'),/SEC|SIC|OSHA/);
assert.match(run('groupDescription({}).caption'),/taxonomy not recorded/);
assert.match(run('groupDescription({taxonomy_metadata:{industry:"observed SEC SIC, zero-padded to four digits",sector:"OSHA SIC division exposed as sector",country:"FSDS countryba business-address country"}}).caption'),/SEC SIC4.*OSHA SIC division.*SEC business address/);
assert.match(run('groupDescription({availability_policy:"after filing",historical_identity_limit:"mapping proxy"}).details.join(" ")'),/after filing.*mapping proxy/);
const engineStart=source.indexOf('function labEngineNote'), engineEnd=source.indexOf('async function useLabSettings',engineStart);
vm.runInContext(source.slice(engineStart,engineEnd),context);
assert.match(run('labEngineNote({engine:"alphalab-1"})'),/assignment and capped-neutrality/);
assert.match(run('labEngineNote({engine:"alphalab-2"})'),/Truncation could break neutrality/);
assert.doesNotMatch(run('labEngineBanner({engine:"alphalab-3"}).class'),/compatibility-note/);
const provenanceStart=source.indexOf('function labClassificationProvenance'), provenanceEnd=source.indexOf('function renderLabDetail',provenanceStart);
vm.runInContext(source.slice(provenanceStart,provenanceEnd),context);
const provenance=run('labClassificationProvenance({summary:{classification_coverage:0.9063652022730565,max_abs_group_net_exposure:5e-17,inactive_neutralized_days:3}}).children.map(child=>child.text).join(" | ")');
assert.match(provenance,/Classification coverage: 90\.6%/);
assert.match(provenance,/Max absolute group net exposure: <0\.000001%/);
assert.match(provenance,/Inactive neutralized days: 3/);
assert.match(provenance,/Classification source: not required/);
assert.match(run('labClassificationProvenance({settings:{neutralization:"INDUSTRY"},dataset:{dataset:"original-prices"}}).children[0].text'),/embedded groups from Trading prices: original-prices/);
assert.match(run('labClassificationProvenance({settings:{neutralization:"NONE"}}).children[0].text'),/not required/);
const preview={path:'classifications/sic',groups:['industry','sector','country'],manifest_sha256:'a',events_sha256:'b'};
const metadata={overlays:[preview],embedded_groups:['market'],groups:['market'],diagnostics:[]};
(async()=>{
  let load=run('loadLabGroups()');
  assert.equal(control('#lab-neutralization').value,'INDUSTRY'); // loading keeps the strategy
  pending.shift().resolve(metadata); await load;
  assert.equal(control('#lab-neutralization').value,'INDUSTRY'); // auto-discovery keeps it too
  assert.equal(run('state.lab.groupOverlay'),preview.path); // sole fresh choice
  run('restoreLabGroup({})'); load=run('loadLabGroups()'); pending.shift().resolve(metadata); await load;
  assert.equal(run('state.lab.groupOverlay'),''); // historic price-only stays empty
  assert.equal(control('#lab-neutralization').value,'INDUSTRY');
  assert.equal(control('#lab-neutralization').options.find(o=>o.value==='INDUSTRY').disabled,true);
  assert.match(control('#lab-neutralization-note').textContent,/needs groups/);
  run('state.lab.groupExplicit=false'); load=run('loadLabGroups()');
  pending.shift().resolve({...metadata,overlays:[preview,{...preview,path:'classifications/other'}]}); await load;
  assert.equal(run('state.lab.groupOverlay'),'');
  assert.match(control('#lab-group-caption').textContent,/Multiple compatible/);
  const old=run('loadLabGroups()'); const oldPending=pending.shift();
  run('state.lab.datasetPath="other-prices"'); const latest=run('loadLabGroups()');
  pending.shift().resolve({...metadata,overlays:[]}); await latest;
  oldPending.resolve(metadata); await old;
  assert.equal(run('state.lab.groupMetadata.overlays.length'),0);
  const staleError=run('loadLabGroups()'), stalePending=pending.shift();
  const success=run('loadLabGroups()'); pending.shift().resolve({...metadata,overlays:[]}); await success;
  stalePending.reject(new Error('old source error')); await staleError;
  assert.equal(run('state.lab.groupMetadata.error'),undefined);
  run('restoreLabGroup({classification:{source_identity:{path:"classifications/sic",manifest_sha256:"saved"}}})');
  load=run('loadLabGroups()'); pending.shift().resolve({...metadata,selected:preview,groups:preview.groups}); await load;
  assert.match(run('state.lab.groupMetadata.error'),/identity differs/);
  assert.equal(run('state.lab.groupOverlay'),preview.path);
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
