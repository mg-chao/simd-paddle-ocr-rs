"""Spatial end-to-end OCR CER against XFUND word annotations (custom, not official XFUND SER scoring)."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics
import unicodedata
import hashlib


def norm(text):
    return ''.join(unicodedata.normalize('NFKC',text).split())

def area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2 if points else 0.

def intersection(points,box):
    """Clip an OCR convex quadrilateral to an axis-aligned annotation rectangle."""
    poly=[tuple(p) for p in points]
    for axis,bound,above in [(0,box[0],True),(0,box[2],False),(1,box[1],True),(1,box[3],False)]:
        old=poly;poly=[]
        if not old: return 0.
        def inside(p): return p[axis]>=bound if above else p[axis]<=bound
        for prev,cur in zip(old[-1:]+old[:-1],old):
            ip,ic=inside(prev),inside(cur)
            if ip != ic:
                t=(bound-prev[axis])/(cur[axis]-prev[axis])
                poly.append((prev[0]+t*(cur[0]-prev[0]),prev[1]+t*(cur[1]-prev[1])))
            if ic: poly.append(cur)
    return area(poly)

def edit_counts(expected,actual):
    """Minimum Levenshtein edits; deterministic tie order substitution, deletion, insertion."""
    n,m=len(expected),len(actual)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=min(dp[i-1][j-1]+(expected[i-1]!=actual[j-1]),dp[i-1][j]+1,dp[i][j-1]+1)
    s=d=ins=0;i=n;j=m
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(expected[i-1]!=actual[j-1]):
            s+=expected[i-1]!=actual[j-1];i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1: d+=1;i-=1
        else: ins+=1;j-=1
    return s,d,ins

def ordered_words(words):
    # Bucket by visual row before x order. Horizontal printed form text dominates XFUND ZH.
    rows=[]
    # Punctuation often has a short, low box. Use the line's median word height
    # so an in-line period is not incorrectly moved behind the entire row.
    row_tolerance=.5*statistics.median(w['box'][3]-w['box'][1] for w in words) if words else 0
    for word in sorted(words,key=lambda w:((w['box'][1]+w['box'][3])/2,w['box'][0])):
        box=word['box'];cy=(box[1]+box[3])/2;h=box[3]-box[1]
        if rows and abs(cy-statistics.median((w['box'][1]+w['box'][3])/2 for w in rows[-1])) <= row_tolerance:
            rows[-1].append(word)
        else: rows.append([word])
    return [w for row in rows for w in sorted(row,key=lambda w:w['box'][0])]

def evaluate(document,output,threshold=.5):
    words=[w for e in document['document'] for w in e['words'] if norm(w['text'])]
    lines=output['lines']; groups=defaultdict(list);missing=[]
    polygons=[l['detection']['points'] for l in lines]
    bounds=[(min(p[0] for p in poly),min(p[1] for p in poly),max(p[0] for p in poly),max(p[1] for p in poly)) for poly in polygons]
    for w in words:
        box=w['box']; size=max(0,(box[2]-box[0])*(box[3]-box[1])); candidates=[]
        for i,b in enumerate(bounds):
            if b[0]>=box[2] or b[2]<=box[0] or b[1]>=box[3] or b[3]<=box[1]: continue
            overlap=intersection(polygons[i],box)/size if size else 0
            if overlap>=threshold: candidates.append((overlap,-area(polygons[i]),-i,i))
        if candidates: groups[max(candidates)[-1]].append(w)
        else: missing.append(w)
    chars=sum(len(norm(w['text'])) for w in words)
    s=ins=0;d=sum(len(norm(w['text'])) for w in missing); exact=0
    for i,line in enumerate(lines):
        expected=''.join(norm(w['text']) for w in ordered_words(groups[i]))
        actual=norm(line['text'])
        si,di,ii=edit_counts(expected,actual);s+=si;d+=di;ins+=ii
        exact += bool(expected) and expected==actual
    matched=chars-s-d
    assert chars>0 and matched>=0
    return {'gt_chars':chars,'substitutions':s,'deletions':d,'insertions':ins,'correct_chars':matched,
        'covered_gt_chars':chars-sum(len(norm(w['text'])) for w in missing),
        'gt_words':len(words),'unassigned_gt_words':len(missing),'predicted_lines':len(lines),'exact_spatial_lines':exact,
        'cer':(s+d+ins)/chars}

def self_test():
    square=[[0,0],[10,0],[10,10],[0,10]]
    assert intersection(square,[2,2,8,8])==36
    assert intersection(square,[20,20,30,30])==0
    assert intersection(square,[5,0,15,10])==50
    assert abs(intersection([[0,5],[5,0],[10,5],[5,10]],[0,0,10,10])-50)<1e-9
    for a,b,counts in [('abc','abc',(0,0,0)),('abc','axc',(1,0,0)),('abc','ac',(0,1,0)),('abc','abxc',(0,0,1)),('','ab',(0,0,2))]:
        assert edit_counts(a,b)==counts
    doc={'document':[{'words':[{'text':'甲','box':[0,0,5,10]},{'text':'乙','box':[5,0,10,10]}]}]}
    out={'lines':[{'text':'甲乙','detection':{'points':square}}]}
    assert evaluate(doc,out)['cer']==0
    assert evaluate(doc,{'lines':[]})['cer']==1
    assert norm('Ａ 甲\n乙')=='A甲乙'
    punct=[{'text':'A','box':[0,0,5,10]},{'text':'.','box':[5,8,6,10]},{'text':'B','box':[6,0,11,10]}]
    assert ''.join(w['text'] for w in ordered_words(punct))=='A.B'


def report_accuracy(report, annotations):
    raw = annotations.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'b11140881255916a16670ea5006295d0185b8376287aa4aa76ae4e1107f61e3d':
        raise ValueError('annotations differ from the frozen XFUND Chinese validation split')
    documents = {d['id']: d for d in json.loads(raw)['documents']}
    manifest = json.loads((report / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('benchmark run is incomplete')
    grouped = defaultdict(dict)
    for run in manifest['runs']:
        if not run['summary']['backend_verified']:
            raise ValueError('unverified backend')
        for row in run['images']:
            if row['id'] == 'sample':
                continue
            key = (run['summary']['preset'], row['phase'])
            if row['id'] in grouped[key]:
                raise ValueError('duplicate corpus prediction')
            grouped[key][row['id']] = row['output']
    if not grouped:
        raise ValueError('no corpus predictions')
    summaries, detail = [], []
    for (preset, phase), outputs in sorted(grouped.items()):
        if outputs.keys() != documents.keys():
            raise ValueError(f'{preset}/{phase}: expected all 50 images')
        scores = []
        for image_id, output in outputs.items():
            score = evaluate(documents[image_id], output)
            scores.append(score)
            detail.append(dict(preset=preset, phase=phase, id=image_id, **score))
        totals = {k: sum(s[k] for s in scores) for k in scores[0] if k != 'cer'}
        cer = sum(totals[k] for k in ('substitutions', 'deletions', 'insertions')) / totals['gt_chars']
        summaries.append(dict(preset=preset, phase=phase, images=len(scores), cer=cer, totals=totals))
    return dict(implementation=manifest['implementation'], annotations_sha256=digest,
                metric='Spatial CER: NFKC, whitespace ignored, case/punctuation retained; GT-word coverage >= 0.5; visual row then x ordering; unassigned GT deletions, unmatched prediction insertions; micro-average. Not official XFUND SER/RE.',
                summary=summaries, per_image=detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--annotations', type=Path, default=Path(__file__).resolve().parents[2] / '.local/datasets/xfund-zh-val/zh.val.json')
    args = parser.parse_args()
    self_test()
    result = report_accuracy(args.report, args.annotations)
    (args.report / 'accuracy.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in result['summary']:
        print(f"{row['preset']} {row['phase']}: CER {100*row['cer']:.4f}% ({row['images']} images)")


if __name__ == '__main__':
    main()
