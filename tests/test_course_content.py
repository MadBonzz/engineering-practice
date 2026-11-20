"""Independent numerical/property checks and exported-artifact regression checks."""
import copy
import importlib
import json
import math
from pathlib import Path
import random
import sys
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'authoring/courses'))
f=importlib.import_module('foundations')
s=importlib.import_module('systems')
l=importlib.import_module('language')
e=importlib.import_module('experiments')


class NumericalOracles(unittest.TestCase):
    def test_classifier_finite_differences(self):
        rng=random.Random(31)
        for _ in range(12):
            x=[[rng.uniform(-2,2) for _ in range(2)] for _ in range(3)]
            weights=[[rng.uniform(-1,1) for _ in range(3)] for _ in range(2)];labels=[0,2,1]
            result=f.classifier_gradient(x,weights,labels)
            def independent_loss(w):
                value=0
                for row,y in zip(x,labels):
                    scores=[sum(row[k]*w[k][j] for k in range(2)) for j in range(3)]
                    value+=math.log(sum(math.exp(v) for v in scores))-scores[y]
                return value/3
            for i in range(2):
                for j in range(3):
                    plus=copy.deepcopy(weights);minus=copy.deepcopy(weights);plus[i][j]+=1e-5;minus[i][j]-=1e-5
                    self.assertAlmostEqual(result['gradient'][i][j],(independent_loss(plus)-independent_loss(minus))/2e-5,places=7)

    def test_shared_graph_against_closed_form(self):
        for x in [-2,-.7,0,.3,2]:
            # tanh(x*x+x) + x*x, with the square shared twice.
            nodes=[{'op':'input','value':x},{'op':'mul','parents':[0,0]},{'op':'add','parents':[1,0]},{'op':'tanh','parents':[2]},{'op':'add','parents':[3,1]}]
            derivative=(1-math.tanh(x*x+x)**2)*(2*x+1)+2*x
            self.assertAlmostEqual(f.reverse_graph(nodes,4,1)[0],derivative)

    def test_view_adjoint(self):
        storage=[.2,-.4,.8,1.2];shape=[2,3];strides=[1,0];upstream=[1,2,-1,3,4,2]
        result=f.view_vjp(storage,shape,strides,1,upstream)
        self.assertAlmostEqual(sum(a*b for a,b in zip(result['values'],upstream)),sum(a*b for a,b in zip(storage,result['storage_gradient'])))

    def test_convolution_finite_differences(self):
        image=[[.1,.4,-.2],[.8,.3,-.5]];weights=[[.3,.8],[-.4,.2]]
        for stride,padding in [(1,0),(1,1),(2,1)]:
            out=f.filter_forward(image,weights,stride,padding);upstream=[[1+row+col for col in range(len(out[0]))] for row in range(len(out))]
            result=f.filter_backward(image,weights,stride,padding,upstream)
            def direct_loss(x,w):
                total=0
                for oy,row in enumerate(upstream):
                    for ox,g in enumerate(row):
                        for y in range(len(x)):
                            for z in range(len(x[0])):
                                ky=y-oy*stride+padding;kx=z-ox*stride+padding
                                if 0<=ky<2 and 0<=kx<2:total+=g*x[y][z]*w[ky][kx]
                return total
            for key,original in [('image_gradient',image),('weight_gradient',weights)]:
                for i,row in enumerate(original):
                    for j in range(len(row)):
                        plus=copy.deepcopy(original);minus=copy.deepcopy(original);plus[i][j]+=1e-5;minus[i][j]-=1e-5
                        value=(direct_loss(plus,weights)-direct_loss(minus,weights)) if key=='image_gradient' else (direct_loss(image,plus)-direct_loss(image,minus))
                        self.assertAlmostEqual(result[key][i][j],value/2e-5,places=7)

    def test_tiny_network_finite_differences(self):
        x=[[1,2],[-1,.5]];y=[.4,-.8];weights={'hidden':[[.1,-.2],[.3,.4]],'readout':[.5,-.6]}
        g=e.tiny_gradient(x,y,weights)
        def loss(w):
            # Evaluate a scalar expression independently of tiny_predict.
            p=[w['readout'][0]*math.tanh(w['hidden'][0][0]*a+w['hidden'][0][1]*b)+w['readout'][1]*math.tanh(w['hidden'][1][0]*a+w['hidden'][1][1]*b) for a,b in x]
            return sum((a-b)**2 for a,b in zip(p,y))/4
        for i in range(2):
            for j in range(2):
                plus=copy.deepcopy(weights);minus=copy.deepcopy(weights);plus['hidden'][i][j]+=1e-5;minus['hidden'][i][j]-=1e-5
                self.assertAlmostEqual(g['hidden'][i][j],(loss(plus)-loss(minus))/2e-5,places=7)
            plus=copy.deepcopy(weights);minus=copy.deepcopy(weights);plus['readout'][i]+=1e-5;minus['readout'][i]-=1e-5
            self.assertAlmostEqual(g['readout'][i],(loss(plus)-loss(minus))/2e-5,places=7)

    def test_validation_never_trains(self):
        a=e.learning_curve(**e.TRAIN)
        b=e.learning_curve(**dict(e.TRAIN,validation_y=[100,-100]))
        self.assertEqual(a['weights'],b['weights'])
        self.assertNotEqual(a['validation'],b['validation'])
        frozen=e.learning_curve(**dict(e.TRAIN,freeze_hidden=True,decay=2))
        self.assertEqual(frozen['weights']['hidden'],e.NN['hidden'])

    def test_unicode_roundtrip(self):
        rng=random.Random(72)
        texts=[''.join(rng.choice('aé🙂\n漢ß') for _ in range(n)) for n in range(15)]
        for budget in [0,1,8,30]:
            result=l.tokenize(texts[:6],texts,budget)
            self.assertEqual(result['decoded'],texts)
            self.assertEqual(result,l.tokenize(texts[:6],texts,budget))

    def test_packing_preserves_scored_tokens(self):
        docs=[[0,1,0],[],[1],[0,0]]
        for block in range(1,9):
            rows=l.pack_windows(docs,2,3,block)
            self.assertEqual(sum(sum(r['mask']) for r in rows),sum(map(len,docs)))
            logits=[[[0]*4 for _ in range(block)] for _ in rows]
            result=l.corpus_loss(docs,2,3,block,logits)
            self.assertAlmostEqual(result['perplexity'],4)

    def test_tiling_and_sharding_match_dense(self):
        rng=random.Random(52)
        for size in range(1,6):
            a=[[rng.randint(-3,3) for _ in range(size+1)] for _ in range(size)]
            b=[[rng.randint(-3,3) for _ in range(3)] for _ in range(size+1)]
            expected=[[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
            for tile in [1,2,4,8]:self.assertEqual(s.tiled_product(a,b,tile),expected)
            ranks=4;weights=list(range(size));gradients=[[rng.randint(-3,3) for _ in weights] for _ in range(ranks)];counts=[1,2,3,4]
            got=s.sharded_step(weights,gradients,counts,.2,4)
            dense=[w-.2*sum(row[i] for row in gradients)/sum(counts) for i,w in enumerate(weights)]
            for actual,expected_value in zip(got['weights'],dense):self.assertAlmostEqual(actual,expected_value)
            self.assertEqual(sum(got['state_bytes']),16*size)


class ExportContracts(unittest.TestCase):
    def test_catalog_and_artifacts(self):
        catalog=json.loads((ROOT/'public/course-new-catalog.json').read_text())
        bank=json.loads((ROOT/'worker/course-cases.json').read_text())
        combined=json.loads((ROOT/'public/course-catalog.json').read_text())
        self.assertEqual(len(catalog),22)
        self.assertEqual(len({r['slug'] for r in combined}),len(combined))
        self.assertEqual(set(bank),{r['slug'] for r in catalog})
        for row in catalog:
            base=ROOT/'public/courses'/row['slug'];exercise=json.loads((base/'exercise.json').read_text())
            self.assertEqual(row['editable'],exercise['editable'])
            self.assertIn('REFLECTION.md',row['editable'])
            self.assertEqual(len(exercise['questions']),2)
            with zipfile.ZipFile(base/'starter.zip') as archive:
                self.assertEqual(set(archive.namelist()),set(exercise['files']))
                for name,text in exercise['files'].items():self.assertEqual(archive.read(name).decode(),text)
                self.assertFalse(any('reference' in p or 'private' in p or 'mutation' in p for p in archive.namelist()))
            self.assertEqual({case['part'] for case in bank[row['slug']]},{1,2,3})
            self.assertGreaterEqual(sum(c['actions'][0]=='gateway.handle' for c in bank[row['slug']]),3)

    def test_static_and_worker_templates_agree(self):
        for name in ['library.html','desk.html']:
            self.assertEqual((ROOT/'web'/name).read_text(),(ROOT/'public'/name).read_text())


if __name__=='__main__':unittest.main()
