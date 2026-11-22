import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf
SR=44100; BPM=72; BEAT=60/BPM; BARS=18; DUR=BARS*4*BEAT
N=int(DUR*SR)+SR
L=np.zeros(N); R=np.zeros(N)
rng=np.random.default_rng(21)
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
def guitar(m,dur,vel=1.0):
    n=int(dur*SR); f=hz(m); p=int(SR/f); buf=rng.uniform(-1,1,p); out=np.zeros(n)
    for i in range(n):
        out[i]=buf[i%p]; buf[i%p]=0.5*(buf[i%p]+buf[(i+1)%p])*0.994
    b,a=sg.butter(1,[300/(SR/2),5000/(SR/2)],'band')
    return sg.lfilter(b,a,out)*np.exp(-np.arange(n)/SR/0.09)*vel*2.2
def flute(m,dur,vel=1.0):
    n=int(dur*SR); t=np.arange(n)/SR; f=hz(m)
    vib=1+0.006*np.sin(2*np.pi*5.5*t)*np.minimum(1,t/0.3)
    ph=2*np.pi*np.cumsum(f*vib)/SR
    x=np.sin(ph)+0.18*np.sin(2*ph)+0.05*np.sin(3*ph)
    br=rng.standard_normal(n); b,a=sg.butter(2,[1500/(SR/2),6000/(SR/2)],'band'); br=sg.lfilter(b,a,br)
    e=np.minimum(1,t/0.06)*np.minimum(1,(n-np.arange(n))/(0.12*SR))
    return (x+0.08*br)*e*vel
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
# harmony: Dm9 | G13 | Cmaj9 | A7b13  (voicings, midi)
CH=[[58,62,65,69],[58,62,64,67],[57,60,64,67],[54,60,65,69]]
ROOT=[43,48,41,38]
KICK,RIM=kick(),rim()
mel_scale=[67,69,70,72,74,76,77,79,81,82,84]
def bt(bar,beat): return (bar*4+beat)*BEAT
for bar in range(BARS):
    c=bar%4; chord=CH[c]; root=ROOT[c]
    intro=bar<2; outro=bar==BARS-1
    # skank on 2 and 4 (epiano chord), organ bubble on 8th offbeats
    for b in (1,3):
        for k,m in enumerate(chord): add(guitar(m+12,0.3,0.16),bt(bar,b)+0.006*k,pan=-0.3)
    if not intro:
        for e in range(8):
            if e%2==1: 
                for m in chord[:3]: add(organ(m,0.16,0.06),bt(bar,e/2),pan=0.3)
    if outro:
        for k,m in enumerate(chord): add(epiano(m,2.5,0.2),bt(bar,0)+0.03*k)
        add(bass(root,2.5,0.6),bt(bar,0)); continue
    # bass: one-drop style, rest on 1
    if not intro or bar==1:
        pat=[(0.5,0,0.5),(1.5,7,0.4),(2,12,0.9),(2.5,10,0.4),(3,7,0.7),(3.5,5 if c!=3 else 4,0.4)]
        if bar%8==7: pat=[(0.5,0,.5),(1,3,.5),(1.5,5,.5),(2,7,.9),(2.75,10,.5),(3.25,12,.6),(3.75,11,.4)]
        for p,iv,v in pat: add(bass(root+iv,0.45*BEAT*2 if v>.6 else 0.4*BEAT,0.55*v+0.25),bt(bar,p))
    # drums
    if not intro:
        add(KICK,bt(bar,2),g=0.9); add(RIM,bt(bar,2),pan=0.1,g=0.45)
        for e in range(8):
            sw=0.06*BEAT if e%2 else 0
            add(hat(open_=(e==7 and bar%2)),bt(bar,e/2)+sw,pan=0.35,g=0.12 if e%2 else 0.2)
        for s in range(16):
            add(shaker(),bt(bar,s/4)+(0.03*BEAT if s%2 else 0),pan=-0.4,g=0.035)
        if bar%4==3: 
            for f,t in enumerate((3.25,3.5,3.75)): add(RIM,bt(bar,t),pan=-0.2+0.2*f,g=0.3)
# melody (vibes) bars 3..17, phrases built from chord tones, call & response
rng2=np.random.default_rng(13)
motifs=[[(0,0,1),(1,2,.5),(1.5,3,.5),(2,4,1.5)],[(0.5,4,.5),(1,3,.5),(1.5,2,1),(2.5,1,.5),(3,0,1)],[(0,5,.75),(0.75,4,.25),(1,3,1),(2,4,.5),(2.5,6,1.5)],[(0,7,1),(1,6,.5),(1.5,5,.5),(2,4,.5),(2.5,3,.5),(3,2,1)]]
def nearest_chord_tone(m,chord):
    cands=[x+12*o for x in chord for o in (-1,0,1)]
    return min(cands,key=lambda x:abs(x-m))
for bar in range(3,BARS-1):
    if bar in (9,): continue  # breathing space
    mot=motifs[(bar-3)%4] if bar<10 else motifs[(bar+1)%4]
    shift=0 if bar<10 else 2
    for p,deg,d in mot:
        m=mel_scale[min(len(mel_scale)-1,deg+shift)]
        if p in (0,2): m=nearest_chord_tone(m,CH[bar%4])
        add(flute(m,d*BEAT*0.95,0.22),bt(bar,p)+rng2.uniform(0,0.012),pan=0.05)
# reverb
def reverb(x):
    n=int(1.6*SR); t=np.arange(n)/SR; ir=rng.standard_normal(n)*np.exp(-t/0.45); ir[0]=0
    b,a=sg.butter(1,3500/(SR/2)); ir=sg.lfilter(b,a,ir); ir/=np.sqrt((ir**2).sum())
    return sg.fftconvolve(x,ir)[:len(x)]
L2=L+0.22*reverb(L); R2=R+0.22*reverb(R)
# warmth: gentle lowpass + tape-ish saturation
b,a=sg.butter(2,11000/(SR/2)); L2=sg.lfilter(b,a,L2); R2=sg.lfilter(b,a,R2)
st=np.stack([L2,R2],1)[:int(60.0*SR)]
st=np.tanh(1.2*st/np.max(np.abs(st)))
fade=int(1.5*SR); st[-fade:]*=np.linspace(1,0,fade)[:,None]; st[:int(0.05*SR)]*=np.linspace(0,1,int(0.05*SR))[:,None]
st=st/np.max(np.abs(st))*0.89
wf.write('sunrise_merge.wav',SR,(st*32767).astype(np.int16))
print('ok',st.shape[0]/SR)
