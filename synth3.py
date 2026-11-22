import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf
SR=44100; BPM=78; BEAT=60/BPM; BARS=20; DUR=BARS*4*BEAT
N=int(DUR*SR)+SR
L=np.zeros(N); R=np.zeros(N)
rng=np.random.default_rng(77)
def hz(m): return 440*2**((m-69)/12)
def add(sig,t,pan=0.0,g=1.0):
    i=int(t*SR); n=min(len(sig),N-i)
    if n<=0: return
    l=np.cos((pan+1)*np.pi/4); r=np.sin((pan+1)*np.pi/4)
    L[i:i+n]+=sig[:n]*g*l; R[i:i+n]+=sig[:n]*g*r
def env(n,a,d,s=0.0,rel=None):
    t=np.arange(n)/SR; e=np.minimum(1,t/max(a,1e-4))*np.exp(-t/d)
    return e*(1-s)+s*np.minimum(1,t/max(a,1e-4))
# instruments
def epiano(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)
    mod=np.sin(2*np.pi*f*t)*(1.8*np.exp(-t/0.25))
    x=np.sin(2*np.pi*f*t+mod)+0.25*np.sin(2*np.pi*2*f*t)*np.exp(-t/0.3)
    return x*env(n,0.004,0.35)*vel*np.minimum(1,(dur*SR-np.arange(n))/(0.01*SR))
def organ(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)
    x=sum(a*np.sin(2*np.pi*f*k*t) for k,a in ((0.5,.5),(1,1),(2,.5),(3,.25)))
    return x*env(n,0.01,0.12)*vel
def bass(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)
    x=np.sin(2*np.pi*f*t)+0.3*np.sin(2*np.pi*2*f*t)+0.08*np.sin(2*np.pi*3*f*t)
    x=np.tanh(1.6*x)
    e=np.minimum(1,t/0.008)*np.exp(-t/1.2)*np.minimum(1,(n-np.arange(n))/(0.03*SR))
    return x*e*vel
def vibe(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)
    x=np.sin(2*np.pi*f*t)+0.2*np.sin(2*np.pi*4*f*t)*np.exp(-t/0.05)+0.12*np.sin(2*np.pi*10*f*t)*np.exp(-t/0.02)
    trem=1+0.25*np.sin(2*np.pi*5.2*t)
    return x*trem*env(n,0.002,0.9)*vel
def kick():
    n=int(0.35*SR); t=np.arange(n)/SR
    f=50+90*np.exp(-t/0.03); ph=2*np.pi*np.cumsum(f)/SR
    return np.sin(ph)*np.exp(-t/0.16)
def rim():
    n=int(0.12*SR); t=np.arange(n)/SR
    x=np.sin(2*np.pi*1700*t)*np.exp(-t/0.012)+0.6*rng.standard_normal(n)*np.exp(-t/0.02)
    b,a=sg.butter(2,[900/(SR/2),6000/(SR/2)],'band'); return sg.lfilter(b,a,x)
def hat(open_=False):
    n=int((0.25 if open_ else 0.05)*SR); t=np.arange(n)/SR
    x=rng.standard_normal(n); b,a=sg.butter(2,7000/(SR/2),'high'); x=sg.lfilter(b,a,x)
    return x*np.exp(-t/(0.09 if open_ else 0.015))
def shaker():
    n=int(0.06*SR); t=np.arange(n)/SR; x=rng.standard_normal(n)
    b,a=sg.butter(2,[4000/(SR/2),9000/(SR/2)],'band'); return sg.lfilter(b,a,x)*np.sin(np.pi*t/0.06)

def melodica(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)*(1+0.004*np.sin(2*np.pi*5.5*t)*np.minimum(1,t/0.3))
    ph=2*np.pi*np.cumsum(f)/SR
    x=sum(np.sin(k*ph)/k**1.15*(1 if k%2 else 0.55) for k in range(1,9))
    x+=0.05*rng.standard_normal(n)*np.exp(-t/0.05)
    b,a=sg.butter(2,2600/(SR/2)); x=sg.lfilter(b,a,x)
    e=np.minimum(1,t/0.035)*(0.85+0.15*np.exp(-t/0.2))*np.minimum(1,(n-np.arange(n))/(0.05*SR))
    return x*e*vel*0.6
# harmony: Fm9 | Bb13 | Ebmaj9 | Dbmaj7#11 ; turnaround C7#9 in bar 8 of each 8
CH=[[60,63,65,67,72],[62,65,67,68,74],[58,62,65,67,70],[60,61,65,67,68]]
C7=[58,63,64,67,70]
ROOT=[41,46,39,37]
KICK,RIM=kick(),rim()
SKL=np.zeros(N); SKR=np.zeros(N)  # skank bus for dub delay
def addsk(sig,t,g=1.0):
    i=int(t*SR); n=min(len(sig),N-i)
    if n>0: SKL[i:i+n]+=sig[:n]*g; SKR[i:i+n]+=sig[:n]*g
def bt(bar,beat): return (bar*4+beat)*BEAT
for bar in range(BARS):
    c=bar%4; chord=CH[c]; root=ROOT[c]
    if bar%8==7: chord=C7; root=36
    intro=bar<2; outro=bar==BARS-1; drop=bar in (12,13)
    for b in (1,3):
        for k,m in enumerate(chord): addsk(epiano(m,0.22,0.17),bt(bar,b)+0.003*k)
    if outro:
        for k,m in enumerate(CH[0]): add(epiano(m,3.0,0.2),bt(bar,0)+0.012*k)
        add(bass(41,3.0,0.6),bt(bar,0)); continue
    if not intro and not drop:
        for e in (1,3,5,7):
            for m in chord[:3]: add(organ(m,0.14,0.05),bt(bar,e/2),pan=0.35)
    # bass: steppers line, root-fifth walk with chromatic approach
    if bar>=1:
        nxt=ROOT[(c+1)%4] if bar%8!=6 else 36
        pat=[(0,0,.9),(1,7,.6),(1.5,12,.4),(2,0,.8),(2.75,10,.5),(3.5,(nxt-root)-1,.55)]
        for p,iv,v in pat: add(bass(root+iv,0.5*BEAT,0.5*v+0.25),bt(bar,p))
    # drums: steppers (kick every beat), rim on 3, swung hats
    if not intro:
        for b in range(4):
            if not drop or b==0: add(KICK,bt(bar,b),g=0.75 if b!=2 else 0.85)
        add(RIM,bt(bar,2),pan=0.1,g=0.45)
        if not drop:
            for e in range(8):
                sw=0.07*BEAT if e%2 else 0
                add(hat(open_=(e==7 and bar%2==1)),bt(bar,e/2)+sw,pan=0.35,g=0.13 if e%2 else 0.18)
            for s in range(16): add(shaker(),bt(bar,s/4)+(0.03*BEAT if s%2 else 0),pan=-0.4,g=0.03)
        if bar%4==3:
            for f,t in enumerate((3.0,3.33,3.66)): add(RIM,bt(bar,t),pan=-0.3+0.3*f,g=0.28)
# melodica melody: two 8-bar phrases (A, A'), minor pentatonic + 9ths
scale=[60,63,65,67,68,70,72,75,77,79]
A=[[(0,4,1.5),(1.5,5,.5),(2,6,1),(3,5,.5),(3.5,4,.5)],
   [(0,3,2),(2.5,2,.5),(3,1,1)],
   [(0.5,2,.5),(1,3,.5),(1.5,4,1),(2.5,6,.5),(3,7,1)],
   [(0,6,3)],
   [(0,4,1),(1,6,.5),(1.5,7,.5),(2,8,1.5),(3.5,7,.5)],
   [(0,6,1),(1,5,.5),(1.5,4,.5),(2,3,2)],
   [(0,2,.5),(.5,3,.5),(1,4,.5),(1.5,3,.5),(2,2,1),(3,1,1)],
   [(0,0,3.5)]]
def nearest(m,chord):
    cands=[x+12*o for x in chord for o in (-1,0,1)]
    return min(cands,key=lambda x:abs(x-m))
MELL=np.zeros(N)
for bar in range(2,BARS-1):
    if bar in (12,13): continue
    ph=A[(bar-2)%8]; up=12 if bar>=14 else 0
    ch=C7 if bar%8==7 else CH[bar%4]
    for p,deg,d in ph:
        m=scale[deg]+up
        if p==0: m=nearest(m,ch)
        sig=melodica(m,d*BEAT*0.95,0.33)
        i=int(bt(bar,p)*SR); n=min(len(sig),N-i)
        if n>0: MELL[i:i+n]+=sig[:n]
# dub delay on skank (dotted 8th) and light delay on melodica
def delay(x,dt,fb,mix,lp=2500):
    d=int(dt*SR); y=x.copy(); b,a=sg.butter(1,lp/(SR/2))
    tap=x.copy()
    for k in range(1,6):
        tap=sg.lfilter(b,a,np.concatenate([np.zeros(d),tap[:-d]]))*fb
        y+=tap*mix
    return y
skd=delay(SKL,0.75*BEAT,0.55,0.8)
L+=skd*0.9; R+=delay(SKR,0.75*BEAT*1.02,0.55,0.8)*0.9
md=delay(MELL,BEAT,0.4,0.6,3000)
L+=md*0.95; R+=delay(MELL,BEAT*1.5,0.35,0.6,3000)*0.95
# soft rain bed
rain=rng.standard_normal(N); b,a=sg.butter(2,[1500/(SR/2),8000/(SR/2)],'band'); rain=sg.lfilter(b,a,rain)
rain*=0.012*(1+0.3*np.sin(2*np.pi*0.07*np.arange(N)/SR))
L+=rain; R+=np.roll(rain,800)
def reverb(x):
    n=int(1.8*SR); t=np.arange(n)/SR; ir=rng.standard_normal(n)*np.exp(-t/0.5); ir[0]=0
    b,a=sg.butter(1,3200/(SR/2)); ir=sg.lfilter(b,a,ir); ir/=np.sqrt((ir**2).sum())
    return sg.fftconvolve(x,ir)[:len(x)]
L2=L+0.2*reverb(L); R2=R+0.2*reverb(R)
b,a=sg.butter(2,10500/(SR/2)); L2=sg.lfilter(b,a,L2); R2=sg.lfilter(b,a,R2)
st=np.stack([L2,R2],1)[:int(60.0*SR)]
st=np.tanh(1.2*st/np.max(np.abs(st)))
fade=int(2.0*SR); st[-fade:]*=np.linspace(1,0,fade)[:,None]; st[:int(0.05*SR)]*=np.linspace(0,1,int(0.05*SR))[:,None]
st=st/np.max(np.abs(st))*0.89
wf.write('rainy_rebase.wav',SR,(st*32767).astype(np.int16))
print('ok',st.shape[0]/SR, 'dur of song', DUR)
