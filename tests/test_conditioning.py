import json,unittest
from iu_cxr.conditioning import plan_report,normalize_report,empty_windows
from iu_cxr.reference.frozen import r4_token_plan


class WordTokenizer:
    bos_token_id=49406
    eos_token_id=49407
    def __init__(self):self.vocabulary={}
    def encode(self,text,**kwargs):
        if kwargs.get('truncation') is not False:
            raise AssertionError('Tokenizer truncation was not explicitly disabled.')
        result=[]
        for word in text.split():
            self.vocabulary.setdefault(word,len(self.vocabulary)+1)
            result.append(self.vocabulary[word])
        return result


class ConditioningTests(unittest.TestCase):
    def test_long_report_retains_all_tokens_and_finding_at_end(self):
        text=' '.join('word'+str(i) for i in range(180))+' cardiomegaly'
        tokenizer=WordTokenizer();plan=plan_report(text,tokenizer)
        recovered=[x for row,n in zip(plan.input_ids,plan.content_lengths) for x in row[1:1+n]]
        self.assertEqual(recovered,tokenizer.encode(text,truncation=False))
        self.assertEqual(plan.content_tokens,181)
        self.assertEqual(plan.windows,3)

    def test_same_packing_as_frozen_study_function(self):
        cases=['Normal cardiac silhouette. No pleural effusion.',
            ' '.join(['Short sentence.']*65), ' '.join(['long']*170)+'. Final finding.',
            'Findings &amp;amp; impression. No acute process.']
        for text in cases:
            with self.subTest(text=text[:24]):
                tok=WordTokenizer();plan=plan_report(text,tok)
                ref=r4_token_plan(normalize_report(text),tok)
                self.assertEqual(plan.input_ids,json.loads(ref['input_id_chunks_json']))
                self.assertEqual(plan.content_lengths,json.loads(ref['chunk_content_lengths_json']))
                self.assertEqual(plan.content_tokens,int(ref['content_token_count']))

    def test_unconditional_sequence_has_same_window_count(self):
        rows=empty_windows(4)
        self.assertEqual(len(rows),4)
        self.assertTrue(all(row==[49406]+[49407]*76 for row in rows))

    def test_window_limit_rejects_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError,'No tokens were silently truncated'):
            plan_report(' '.join(['token']*151),WordTokenizer(),max_windows=2)

    def test_empty_input_rejected(self):
        for value in ['', '  ',None]:
            with self.assertRaises(ValueError):plan_report(value,WordTokenizer())

    def test_html_normalization_matches_two_pass_policy(self):
        self.assertEqual(normalize_report('A &amp;amp; B'),'A & B')

    def test_wrong_tokenizer_rejected(self):
        tok=WordTokenizer();tok.bos_token_id=0
        with self.assertRaises(ValueError):plan_report('Normal study.',tok)


if __name__=='__main__':unittest.main()

