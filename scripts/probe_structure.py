import os,json,pathlib
root=pathlib.Path("/storage/data/metaiot_data/public_dataset")
out={}
def listing(p,n=12):
    p=pathlib.Path(p)
    try:
        names=sorted(os.listdir(p)); return {"path":str(p),"count":len(names),"names":names[:n]}
    except Exception as e:return {"path":str(p),"error":repr(e)}
paths=[
root/"coloradar/12_21_2020_arpg_lab_run0/single_chip/pointclouds",
root/"coloradar/12_21_2020_arpg_lab_run0/groundtruth",
root/"coloradar/calib",root/"coloradar/calib/transforms",
root/"KRadar",root/"KRadar/1/time_info",root/"KRadar/1/info_label",
root/"KRadar/1/radar_tesseract",root/"KRadar/resources",
root/"RaDelft/Scene1/rosDS/radar_ososos",root/"RaDelft/Scene1/RadarCubes",
root/"RaDelft/Scene1/RawData",
pathlib.Path("/home/metaiot_guest/Workspace/qinzx/KRadar/randt_preprocess_v1"),
pathlib.Path("/home/metaiot_guest/Workspace/qinzx/KRadar/randt_input_v1"),
]
for p in paths:out[str(p)]=listing(p)
for p in [root/"RaDelft/Scene1/rosDS/gps_odom.topic.json", root/"RaDelft/Scene1/rosDS/radar.topic.json",root/"RaDelft/Scene1/rosDS/tf.topic.json"]:
    try:
        with open(p) as f: out[str(p)]={"prefix":f.read(3500)}
    except Exception as e:out[str(p)]={"error":repr(e)}
p=pathlib.Path("/home/metaiot_guest/Workspace/qinzx/milliEgo/reports/structure_probe.json")
p.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

