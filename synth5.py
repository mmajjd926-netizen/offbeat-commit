import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf
SR=44100; BPM=80; BEAT=60/BPM; BARS=20; DUR=BARS*4*BEAT
N=int(DUR*SR)+SR
L=np.zeros(N); R=np.zeros(N)
rng=np.random.default_rng(505)
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

def guitar(m,dur,vel=1.0):
    n=int(dur*SR); f=hz(m); p=int(SR/f); buf=rng.uniform(-1,1,p); out=np.zeros(n)
    for i in range(n):
        out[i]=buf[i%p]; buf[i%p]=0.5*(buf[i%p]+buf[(i+1)%p])*0.994
    b,a=sg.butter(1,[300/(SR/2),5000/(SR/2)],'band')
    return sg.lfilter(b,a,out)*np.exp(-np.arange(n)/SR/0.09)*vel*2.2

def melodica(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)*(1+0.004*np.sin(2*np.pi*5.5*t)*np.minimum(1,t/0.3))
    ph=2*np.pi*np.cumsum(f)/SR
    x=sum(np.sin(k*ph)/k**1.15*(1 if k%2 else 0.55) for k in range(1,9))
    x+=0.05*rng.standard_normal(n)*np.exp(-t/0.05)
    b,a=sg.butter(2,2600/(SR/2)); x=sg.lfilter(b,a,x)
    e=np.minimum(1,t/0.035)*(0.85+0.15*np.exp(-t/0.2))*np.minimum(1,(n-np.arange(n))/(0.05*SR))
    return x*e*vel*0.6

def trumpet(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR
    vib=1+0.006*np.sin(2*np.pi*5.2*t)*np.clip((t-0.18)/0.3,0,1)
    f=hz(m)*vib*(1-0.012*np.exp(-t/0.03)); ph=2*np.pi*np.cumsum(f)/SR
    bright=0.35+0.65*np.minimum(1,t/0.06)
    x=sum(np.sin(k*ph)*(bright**(k-1))/k**0.9 for k in range(1,14))
    x+=0.04*rng.standard_normal(n)*np.exp(-t/0.04)
    b,a=sg.butter(2,[700/(SR/2),3200/(SR/2)],'band'); xm=sg.lfilter(b,a,x)   # harmon-mute colour
    b2,a2=sg.butter(2,1800/(SR/2)); x=0.65*xm+0.35*sg.lfilter(b2,a2,x)
    e=np.minimum(1,t/0.03)*(0.8+0.2*np.exp(-t/0.15))*np.minimum(1,(n-np.arange(n))/(0.06*SR))
    return x*e*vel*0.55
def clav(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)
    x=sg.sawtooth(2*np.pi*f*t,0.3)+0.4*np.sin(2*np.pi*2*f*t)
    b,a=sg.butter(2,[400/(SR/2),3500/(SR/2)],'band'); x=sg.lfilter(b,a,x)
    return x*np.exp(-t/0.07)*np.minimum(1,t/0.002)*vel
# harmony: Bbmaj9 | Gm11 | Cm9 | F13 ; turnaround Dm7 G7#9 Cm7 F7 in the last bar of each 8 (Bb major)
CH=[[57,62,65,69,72],[58,62,65,67,72],[58,62,63,67,70],[57,60,63,65,69]]
ROOT=[46,43,48,41]
TURN=[[57,60,62,65],[59,62,65,70]]  # Dm7 (half bar) -> G7#9-ish
KICK,RIM=kick(),rim()
SKL=np.zeros(N); SKR=np.zeros(N)
def addsk(sig,t,g=1.0):
    i=int(t*SR); n=min(len(sig),N-i)
    if n>0: SKL[i:i+n]+=sig[:n]*g; SKR[i:i+n]+=sig[:n]*g
def bt(bar,beat): return (bar*4+beat)*BEAT
for bar in range(BARS):
    c=bar%4; chord=CH[c]; root=ROOT[c]
    turn=bar%8==7
    intro=bar<1; outro=bar==BARS-1; drop=bar in (13,14)
    if outro:
        for k,m in enumerate(CH[0]): add(epiano(m,3.2,0.22),bt(bar,0)+0.014*k)
        add(bass(46,3.0,0.6),bt(bar,0)); add(trumpet(74,2.8,0.5),bt(bar,0.5),pan=-0.15); continue
    for b in (1,3):
        ch=chord if not turn else TURN[0 if b==1 else 1]
        for k,m in enumerate(ch): addsk(epiano(m,0.2,0.13),bt(bar,b)+0.004*k)
        for k,m in enumerate(ch[:3]): addsk(clav(m+12,0.12,0.06),bt(bar,b)+0.002*k)
    if not intro and not drop:
        for e in (1,3,5,7):
            for m in chord[:3]: add(organ(m,0.13,0.045),bt(bar,e/2),pan=0.35)
    # bass: one-drop style, space on beat 1, melodic line; walking in bars 15-18
    if bar>=1:
        nxt=ROOT[(c+1)%4]
        if 15<=bar<=18:
            pat=[(0,0,.8),(1,4 if c in (0,) else 3,.6),(2,7,.7),(3,(nxt-root)+1 if (nxt-root)<6 else (nxt-root)-1,.6)]
        else:
            pat=[(0.5,0,.8),(1.5,7,.55),(2,12,.5),(2.5,10,.5),(3,7,.6),(3.5,5,.45)]
        for p,iv,v in pat: add(bass(root+iv,0.45*BEAT,0.5*v+0.25),bt(bar,p))
    # drums: one-drop (kick+rim on 3), cross-stick on 2 & 4 ghosted, swung hats
    if not intro:
        if not drop: add(KICK,bt(bar,2),g=0.85)
        add(RIM,bt(bar,2),pan=0.1,g=0.5)
        for b in (1,3): add(RIM,bt(bar,b)+0.5*BEAT,pan=-0.2,g=0.1)
        if not drop:
            for e in range(8):
                sw=0.09*BEAT if e%2 else 0
                add(hat(open_=(e==5 and bar%4==1)),bt(bar,e/2)+sw,pan=0.35,g=0.11 if e%2 else 0.16)
            for s in range(16): add(shaker(),bt(bar,s/4)+(0.035*BEAT if s%2 else 0),pan=-0.45,g=0.028)
        if bar%8==7:
            for f,t in enumerate((3.0,3.25,3.5,3.75)): add(RIM,bt(bar,t),pan=-0.3+0.2*f,g=0.22+0.04*f)
# muted-trumpet melody (Bb major pentatonic + 9/6), two 8-bar phrases, second one an octave-free variation
scale=[58,60,62,65,67,70,72,74,77,79]
A=[[(0.5,3,.5),(1,4,1),(2,5,.5),(2.5,4,.5),(3,3,1)],
   [(0,2,1.5),(1.5,1,.5),(2,2,2)],
   [(0,4,.5),(.5,5,.5),(1,6,1),(2,5,.5),(2.5,4,.5),(3,5,1)],
   [(0,3,3)],
   [(0.5,5,.5),(1,6,.5),(1.5,7,1),(2.5,6,.5),(3,5,1)],
   [(0,4,1),(1,3,.5),(1.5,2,.5),(2,3,1.5)],
   [(0,1,.5),(.5,2,.5),(1,3,1),(2,4,.5),(2.5,3,.5),(3,2,.5),(3.5,1,.5)],
   [(0,2,3.5)]]
B=[[(0,7,1),(1,6,.5),(1.5,5,.5),(2,6,1.5)],
   [(0,5,.5),(.5,4,.5),(1,3,1),(2.5,4,.5),(3,5,1)],
   [(0,6,1.5),(1.5,7,.5),(2,8,1),(3,7,1)],
   [(0,6,2),(2.5,5,.5),(3,4,1)],
   [(0,3,.5),(.5,4,.5),(1,5,.5),(1.5,6,.5),(2,7,2)],
   [(0,6,1),(1,5,1),(2,4,2)],
   [(0,2,.5),(.5,3,.5),(1,4,.5),(1.5,3,.5),(2,2,1),(3,1,.5),(3.5,0,.5)],
   [(0,2,3.5)]]
def nearest(m,chord):
    cands=[x+12*o for x in chord for o in (-1,0,1)]
    return min(cands,key=lambda x:abs(x-m))
MEL=np.zeros(N)
for bar in range(2,BARS-1):
    if bar in (13,14): continue
    ph=(A if bar<10 else B)[(bar-2)%8]
    ch=CH[bar%4]
    for p,deg,d in ph:
        m=scale[deg]
        if p==0: m=nearest(m,ch+[x+12 for x in ch])
        sig=trumpet(m,d*BEAT*0.92,0.9)
        i=int(bt(bar,p)*SR); n=min(len(sig),N-i)
        if n>0: MEL[i:i+n]+=sig[:n]
# drop bars 13-14: clav + trumpet echo only (dub break)
for bar in (13,14):
    for p,deg,d in ((0,7,.5),(1.5,5,.5),(3,3,.5)):
        sig=trumpet(scale[deg],d*BEAT,0.6); i=int(bt(bar,p)*SR); n=min(len(sig),N-i); MEL[i:i+n]+=sig[:n]
def delay(x,dt,fb,mix,lp=2500):
    d=int(dt*SR); y=x.copy(); b,a=sg.butter(1,lp/(SR/2))
    tap=x.copy()
    for k in range(1,6):
        tap=sg.lfilter(b,a,np.concatenate([np.zeros(d),tap[:-d]]))*fb
        y+=tap*mix
    return y
L+=delay(SKL,0.75*BEAT,0.45,0.6)*0.85; R+=delay(SKR,0.75*BEAT*1.02,0.45,0.6)*0.85
L+=delay(MEL,0.75*BEAT,0.3,0.45,2800)*1.45; R+=delay(MEL,BEAT,0.3,0.45,2800)*1.4
# train rhythm bed: soft rail clicks every half bar, low rumble
n=N; rum=rng.standard_normal(n); b,a=sg.butter(2,180/(SR/2)); rum=sg.lfilter(b,a,rum)*0.05
L+=rum; R+=rum
for bar in range(BARS-1):
    for p in (0,0.25,2,2.25):
        k=int(0.05*SR); t=np.arange(k)/SR; cl=rng.standard_normal(k)*np.exp(-t/0.008)
        b,a=sg.butter(2,[800/(SR/2),3000/(SR/2)],'band'); add(sg.lfilter(b,a,cl),bt(bar,p),pan=-0.6,g=0.035)
def reverb(x):
    n=int(1.6*SR); t=np.arange(n)/SR; ir=rng.standard_normal(n)*np.exp(-t/0.45); ir[0]=0
    b,a=sg.butter(1,3400/(SR/2)); ir=sg.lfilter(b,a,ir); ir/=np.sqrt((ir**2).sum())
    return sg.fftconvolve(x,ir)[:len(x)]
L2=L+0.22*reverb(L); R2=R+0.22*reverb(R)
b,a=sg.butter(2,10500/(SR/2)); L2=sg.lfilter(b,a,L2); R2=sg.lfilter(b,a,R2)
st=np.stack([L2,R2],1)[:int(60.0*SR)]
st=np.tanh(1.2*st/np.max(np.abs(st)))
fade=int(2.0*SR); st[-fade:]*=np.linspace(1,0,fade)[:,None]; st[:int(0.05*SR)]*=np.linspace(0,1,int(0.05*SR))[:,None]
st=st/np.max(np.abs(st))*0.89
wf.write('platform_two.wav',SR,(st*32767).astype(np.int16))
print('ok',st.shape[0]/SR,'song',DUR)
