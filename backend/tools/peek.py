import sys, numpy as np, trimesh, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fs=sys.argv[1:-1]; out=sys.argv[-1]
fig,ax=plt.subplots(len(fs),3,figsize=(15,5*len(fs)))
for i,f in enumerate(fs):
    m=trimesh.load(f,force='mesh'); v=m.vertices
    if m.extents.max()<5: v=v*1000
    c=v-v.mean(0); _,_,vt=np.linalg.svd(c,full_matrices=False); p=c@vt.T
    for j,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
        ax[i,j].scatter(p[:,a],p[:,b],s=0.2,c=p[:,3-a-b],cmap='viridis'); ax[i,j].set_aspect('equal'); ax[i,j].set_title(f"{f.split('/')[-1]} pc{a}-pc{b} mm")
    print(f, 'boundaries:', len(m.outline().entities) if not m.is_watertight else 0)
plt.savefig(out,dpi=70)
