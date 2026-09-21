import json, numpy as np
from pathlib import Path
files={k:np.load(v) for k,v in {"original":"original.npz","candidate_dense":"candidate_dense.npz","candidate_compact":"candidate_compact.npz"}.items()}
open_fields=(1,3,4,6,7,9,10,11,13,14); allowance=2.5e-4
rows=[]
for i in range(19):
 key=f"field_{i:02d}"; o=files["original"][key]
 rec={"index":i,"shape":list(o.shape),"dtype":str(o.dtype)}
 for n in ("candidate_dense","candidate_compact"):
  x=files[n][key]; d=np.abs(x.astype(np.float64)-o.astype(np.float64)); rec[n]={"bitwise_equal":bool(np.array_equal(x.view(np.uint32),o.view(np.uint32))),"max_abs":float(d.max()),"mismatch_count":int(np.count_nonzero(x.view(np.uint32)!=o.view(np.uint32)))}
 if i<15 or i==18:
  rec["finite"]=bool(np.isfinite(o).all());rec["min"]=float(o.min());rec["max"]=float(o.max());rec["bounded_pass"]=bool(o.min()>=-allowance and o.max()<=1+allowance)
 if i in open_fields:
  e=np.abs(o.astype(np.float64)-1.0);rec["unity"]={"pass":bool(np.all(e<=allowance)),"max_abs":float(e.max()),"violation_count":int(np.count_nonzero(e>allowance)),"unique":[float(x) for x in np.unique(o)]}
 rows.append(rec)
summary={"output_count":19,"field_count":15,"visibility_indices":[15,16,17],"svftotal_index":18,"original_vs_candidate_dense_all_bitwise":all(r["candidate_dense"]["bitwise_equal"] for r in rows),"original_vs_candidate_compact_all_bitwise":all(r["candidate_compact"]["bitwise_equal"] for r in rows),"all_bounded":all(r.get("bounded_pass",True) for r in rows),"unity_failed_indices":[r["index"] for r in rows if "unity" in r and not r["unity"]["pass"]],"fields":rows}
Path("comparison.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:summary[k] for k in ("original_vs_candidate_dense_all_bitwise","original_vs_candidate_compact_all_bitwise","all_bounded","unity_failed_indices")},indent=2))
