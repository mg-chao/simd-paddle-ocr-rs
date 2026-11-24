"""Fixed corpus metrics and deterministic greedy convex-polygon matching."""
import json
import sys
from collections import Counter
from math import dist
from pathlib import Path
from statistics import mean
from shapely import Polygon, STRtree

OUT = Path(__file__).resolve().parents[2] / "reports/rapidocr-parity"


def edit_distance(a, b):
    # Myers' bit-vector Levenshtein algorithm on Unicode code points.
    if not a:
        return len(b)
    masks = {}
    for i, char in enumerate(a):
        masks[char] = masks.get(char, 0) | (1 << i)
    score, high, pos, neg = len(a), 1 << (len(a)-1), -1, 0
    for char in b:
        eq = masks.get(char, 0)
        xv = eq | neg
        xh = (((eq & pos) + pos) ^ pos) | eq
        ph = neg | ~(xh | pos)
        mh = pos & xh
        score += bool(ph & high) - bool(mh & high)
        ph = (ph << 1) | 1
        pos = (mh << 1) | ~(xv | ph)
        neg = ph & xv
    return score


def match(ref, pred):
    a, b = [[Polygon(x['box']).convex_hull for x in lines] for lines in [ref, pred]]
    tree = STRtree(b)
    edges = []
    for i, p in enumerate(a):
        for j in tree.query(p):
            area = p.intersection(b[j]).area
            total = p.area + b[j].area - area
            if total > 0 and area/total >= 0.5:
                edges.append((-area/total, i, int(j)))
    used_a, used_b, matched = set(), set(), []
    for neg_iou, i, j in sorted(edges):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append(dict(ref=i, pred=j, iou=-neg_iou, delta=pred[j]['score']-ref[i]['score'],
                            corner_px=mean(dist(u, v) for u, v in zip(ref[i]['box'], pred[j]['box'], strict=True)),
                            same=pred[j]['text']==ref[i]['text']))
    return sorted(matched, key=lambda x: x['ref'])


def assess(ref, pred):
    a,b = [[l['text'] for l in ls] for ls in (ref,pred)]
    ra, rb = '\n'.join(a), '\n'.join(b)
    links = match(ref,pred)
    return dict(exact=a==b, edits=edit_distance(ra,rb), chars=len(ra), matches=links,
                missing=len(a)-len(links),extra=len(b)-len(links),
                ref_lines=len(a), pred_lines=len(b), empty_ref=not a, empty_pred=not b,
                inversions=sum(l['pred'] > r['pred'] for k, l in enumerate(links) for r in links[k+1:]),
                reordered=any(q['pred']>r['pred'] for q,r in zip(links,links[1:])),
                blank=sum(not t.strip() for t in b))


def main(lhs, rhs, partial=False):
    details = {}
    errors = {}
    for row in json.loads((OUT/'inputs.json').read_text('utf-8')):
        name = row['path']
        files = [OUT/k/(name+'.json') for k in (lhs,rhs)]
        if not all(f.exists() for f in files) and partial:
            continue
        data = [json.loads(f.read_text('utf-8')) for f in files]
        assert all(d['path']==name and d['source_sha256']==row['sha256'] for d in data)
        if any(d['status'] != 'ok' for d in data):
            errors[name] = [d.get('error') for d in data]
        else:
            details[name] = assess(*(d['lines'] for d in data))
    sums = Counter()
    links = []
    for d in details.values():
        sums.update({k:int(v) for k,v in d.items() if k!='matches'})
        links.extend(d['matches'])
    summary = dict(images=len(details)+len(errors), errors=errors, **sums, matched=len(links),
                   cer=sums['edits']/max(1,sums['chars']),
                   iou=mean(x['iou'] for x in links) if links else None,
                   corner_px=mean(x['corner_px'] for x in links) if links else None,
                   score_mae=mean(abs(x['delta']) for x in links) if links else None,
                   different_text=sum(not x['same'] for x in links))
    print(json.dumps(summary, ensure_ascii=True))
    (OUT/(lhs.replace('/', '-')+'-vs-'+rhs.replace('/', '-')+'.json')).write_text(json.dumps(dict(summary=summary,images=details),
                                                        ensure_ascii=False),encoding='utf-8')
    return details


if __name__ == '__main__':
    main(*sys.argv[1:])
