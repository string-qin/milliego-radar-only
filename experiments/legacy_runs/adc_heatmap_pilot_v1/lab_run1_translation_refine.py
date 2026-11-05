from pathlib import Path
src=Path('runs/adc_heatmap_pilot_v1/lab_run1_gap2_holdout.py').read_text()
src=src.replace("pilot/'lab_run1_gap2_holdout'","pilot/'lab_run1_translation_refine'")
src=src.replace("cache=holdout/'spectra.npz'","cache=pilot/'lab_run1_gap2_holdout/spectra.npz'\nassert cache.exists()")
src=src.replace(" for arm in ('joint','yaw_warmup'):"," for arm in ('yaw_warmup','translation_refine'):")
a=src.index('  torch.manual_seed(seed)')
b=src.index('  m.eval()',a)
block=src[a:b].replace("if arm=='yaw_warmup' and step<560","if step<560")
src=src[:a]+"  if arm=='yaw_warmup':\n"+''.join(' '+line for line in block.splitlines(keepends=True))+"""  else:
   m.eval()
   m.requires_grad_(False)
   m.translation.requires_grad_(True)
   opt=torch.optim.RMSprop(m.translation.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
   with torch.inference_mode():
    cnn=m.cnn
    z=torch.cat([cnn(v[:,0])[:,None] for v in x.split(32)])
   z=z.clone()
   for step in range(560):
    ix=torch.randint(len(x),(24,),device='cuda')
    p=m.forward_features(z[ix])[:,0,:3]
    l=(p-y[ix,:3]).square().mean()
    assert torch.isfinite(l)
    opt.zero_grad(set_to_none=True);l.backward();opt.step()
"""+src[b:]
src=src.replace("if arm=='joint':","if arm=='yaw_warmup':")
src=src.replace("  met,trajs=metrics", """  if arm=='yaw_warmup': original_rotation=p[:,3:].copy()
  else: np.testing.assert_array_equal(original_rotation,p[:,3:])
  met,trajs=metrics""")
exec(compile(src,'lab_run1_translation_refine','exec'))

