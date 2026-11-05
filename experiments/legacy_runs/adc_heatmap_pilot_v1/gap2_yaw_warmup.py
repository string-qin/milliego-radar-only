from pathlib import Path
src=Path('runs/adc_heatmap_pilot_v1/gap2_diagnostic.py').read_text()
prefix=src[:src.index('results=[]')]
body=src[src.index('results=[]'):]
body=body.replace("('native','gap2')","('joint','yaw_warmup')").replace("raw_train[k] if arm=='native' else long_train","long_train").replace("for _ in range(1120):","for step in range(1120):").replace("p=m(x[ix])[:,0]; l=loss(p,y[ix]); assert torch.isfinite(l)","p=m(x[ix])[:,0]; l=((p[:,5]-y[ix,5])**2).mean() if arm=='yaw_warmup' and step<560 else loss(p,y[ix]); assert torch.isfinite(l)")
body=body.replace("   opt.zero_grad(set_to_none=True);l.backward();opt.step()","   if step==560:\n    m.eval()\n    with torch.inference_mode(): mid=torch.cat([m(b)[:,0,5].cpu() for b in make(long_test).split(32)]).numpy()\n    print('mid',seed,arm,float(np.sqrt(np.mean((mid-truth[:,5])**2))),flush=True)\n    m.train()\n   opt.zero_grad(set_to_none=True);l.backward();opt.step()")
body=body.replace("make(short_test if arm=='native' else long_test)","make(long_test)")
start=body.index("  if arm=='native':")
end=body.index("  met,_=",start)
body=body[:start]+body[end:]
body=body.replace("gap2_diagnostic.json","gap2_yaw_warmup_results.json").replace("ate=met['ate_initial_rmse_m']","mid_yaw=float(np.sqrt(np.mean((mid-truth[:,5])**2))),translation=met['pair_translation_rmse_m'],ate=met['ate_initial_rmse_m']")
exec(compile(prefix+body,'gap2_yaw_warmup','exec'))

