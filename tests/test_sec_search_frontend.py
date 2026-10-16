"""Search completeness and run-index preparation in the packaged frontend."""
import shutil
import unittest

try:
    from . import test_sec_dictionary_frontend as frontend
except ImportError:
    import test_sec_dictionary_frontend as frontend


@unittest.skipUnless(shutil.which('node'), 'Node is needed for frontend behavior')
class SearchFrontendTests(unittest.TestCase):
    js = frontend.SecDictionaryFrontendTests.js

    def test_unknown_total_and_partial_window_use_explicit_pagination(self):
        result = self.js('''
          api=async()=>({status:'ready',total:null,items:[item],truncated:true,
            has_more:true,matched_sample:100,message:'Observed recipes available'});
          await searchSecDictionary(true);
          return {page:$('#sec-dictionary-page').textContent,
            status:$('#sec-dictionary-status').textContent,next:$('#sec-dictionary-next').disabled};
        ''')
        self.assertIn('total unknown',result['page'])
        self.assertIn('partial matches',result['page'])
        self.assertIn('Partial search results',result['status'])
        self.assertFalse(result['next'])

    def test_timeout_has_more_null_does_not_enable_next_or_claim_zero(self):
        result = self.js('''
          api=async()=>({status:'ready',total:null,items:[],truncated:true,has_more:null});
          await searchSecDictionary(true);
          return {page:$('#sec-dictionary-page').textContent,next:$('#sec-dictionary-next').disabled};
        ''')
        self.assertEqual(result['page'],
                         '0–0 · total unknown starter concepts · partial matches, not global counts')
        self.assertNotIn('of 0', result['page'])
        self.assertTrue(result['next'])

    def test_browse_unknown_count_can_still_page(self):
        result = self.js('''
          api=async()=>({status:'ready',total:null,items:[item],has_more:true,truncated:false});
          await searchSecDictionary(true);
          return {page:$('#sec-dictionary-page').textContent,next:$('#sec-dictionary-next').disabled};
        ''')
        self.assertNotIn('partial',result['page'])
        self.assertFalse(result['next'])

    def test_dictionary_ready_execution_missing_can_prepare_same_job(self):
        result = self.js('''
          api=async()=>({status:'ready',items:[item],total:1,message:'Dictionary ready',
            execution:{status:'missing',message:'Run index must be prepared'}});
          await searchSecDictionary(true);
          const before={hidden:$('#sec-dictionary-build').hidden,
            label:$('#sec-dictionary-build').textContent,status:$('#sec-dictionary-status').textContent};
          api=async(name,data)=>{calls.push({name,data});return {id:'job'};};
          startPolling=job=>calls.push({name:'poll',job});
          await buildSecDictionary();await buildSecDictionary();
          return {before,calls};
        ''')
        self.assertFalse(result['before']['hidden'])
        self.assertEqual(result['before']['label'],'Prepare run index')
        self.assertIn('Dictionary ready Run index must be prepared',result['before']['status'])
        self.assertEqual([c['name'] for c in result['calls']],['sec-dictionary-build','poll'])

    def test_execution_ready_and_absent_legacy_metadata_hide_build(self):
        result = self.js('''
          const outputs=[];
          for(const execution of [{status:'ready'},{},undefined]){
            api=async()=>({status:'ready',items:[item],total:1,execution});
            await searchSecDictionary(true); await buildSecDictionary();
            outputs.push({hidden:$('#sec-dictionary-build').hidden,calls:[...calls]});
          }
          return outputs;
        ''')
        self.assertTrue(all(r['hidden'] and not r['calls'] for r in result))


if __name__ == '__main__':
    unittest.main()
