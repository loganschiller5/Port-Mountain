# ================================================================
# EQUITYLENS — 3D Mountain Portfolio Dashboard
# FIN 330 Final Project
# ================================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(
    page_title="EquityLens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "view" not in st.session_state:
    st.session_state.view = "home"
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "section" not in st.session_state:
    st.session_state.section = None

PORT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]
PORT_WEIGHTS = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.20, "NVDA": 0.20, "JPM": 0.10}
TICKER_COLORS = {
    "AAPL": "#60a5fa", "MSFT": "#34d399", "GOOGL": "#f59e0b",
    "NVDA": "#a78bfa", "JPM": "#fb7185"
}
PALETTE = ["#60a5fa","#34d399","#f59e0b","#a78bfa","#fb7185"]

SECTIONS = {
    "performance": {"label": "Performance", "icon": "📈", "peak": "summit"},
    "allocation":  {"label": "Allocation",  "icon": "🥧", "peak": "ridge"},
    "risk":        {"label": "Risk",        "icon": "⚡", "peak": "face"},
    "holdings":    {"label": "Holdings",    "icon": "💎", "peak": "base"},
}

# ────────────────────────────────────────────
# DATA
# ────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_prices(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.index = pd.to_datetime(df.index)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_info(ticker):
    try: return yf.Ticker(ticker).info or {}
    except: return {}

@st.cache_data(ttl=300, show_spinner=False)
def get_quick(ticker):
    df = get_prices(ticker, "1mo")
    if df.empty: return None, None, None
    c = df["Close"].squeeze()
    return float(c.iloc[-1]), float((c.iloc[-1]-c.iloc[0])/c.iloc[0]), c

@st.cache_data(ttl=300, show_spinner=False)
def port_calc(tickers, weights_t, period):
    frames = {}
    for t in tickers:
        df = get_prices(t, period)
        if not df.empty: frames[t] = df["Close"].squeeze()
    if not frames: return None
    px = pd.DataFrame(frames).dropna()
    rets = px.pct_change().dropna()
    wd = dict(zip(tickers, weights_t))
    w = np.array([wd.get(t,0) for t in px.columns], dtype=float); w /= w.sum()
    pr = rets.dot(w)
    spy = get_prices("SPY", period)
    sr = spy["Close"].squeeze().pct_change().dropna() if not spy.empty else None
    total = float((1+pr).prod()-1); av = float(pr.std()*np.sqrt(252))
    sharpe = float((pr.mean()*252)/av) if av>0 else 0.0
    bt = float((1+sr).prod()-1) if sr is not None else 0.0
    cum = (1+pr).cumprod(); bc = (1+sr).cumprod() if sr is not None else None
    dd = (cum-cum.cummax())/cum.cummax()*100
    ind_r = {}
    for t in tickers:
        dft = get_prices(t, period)
        if not dft.empty:
            c = dft["Close"].squeeze()
            ind_r[t] = float((c.iloc[-1]/c.iloc[0]-1)*100)
    return dict(total=total, av=av, sharpe=sharpe, bt=bt,
                outperf=total-bt, cum=cum, bc=bc, rets=rets,
                pr=pr, dd=dd, ind_r=ind_r)

# ────────────────────────────────────────────
# INDICATORS
# ────────────────────────────────────────────
def calc_rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+up/dn.replace(0,np.nan))).rename("RSI")

def calc_macd(s):
    m=(s.ewm(span=12,adjust=False).mean()-s.ewm(span=26,adjust=False).mean()).rename("MACD")
    sig=m.ewm(span=9,adjust=False).mean().rename("Signal")
    return m, sig, (m-sig).rename("Hist")

def calc_bb(s, n=20):
    mid=s.rolling(n).mean(); std=s.rolling(n).std()
    return (mid+2*std).rename("U"), mid.rename("M"), (mid-2*std).rename("L")

def ann_vol(s):
    return float(s.pct_change().dropna().rolling(20).std().iloc[-1]*np.sqrt(252)*100)

def trend_sig(df):
    c=df["Close"].squeeze(); px=float(c.iloc[-1])
    m20=float(c.rolling(20).mean().iloc[-1]); m50=float(c.rolling(50).mean().iloc[-1])
    if px>m20>m50: lbl,col="Strong Uptrend","#10b981"
    elif px<m20<m50: lbl,col="Strong Downtrend","#ef4444"
    else: lbl,col="Mixed / Sideways","#f59e0b"
    return dict(px=px,m20=m20,m50=m50,lbl=lbl,col=col)

def mom_sig(df):
    r=calc_rsi(df["Close"].squeeze()); rv=float(r.iloc[-1])
    if rv>70: lbl,col="Overbought","#ef4444"
    elif rv<30: lbl,col="Oversold","#10b981"
    else: lbl,col="Neutral","#64748b"
    return dict(val=rv,lbl=lbl,col=col,series=r)

def vol_sig(df):
    v=ann_vol(df["Close"].squeeze())
    if v>40: lbl,col="High","#ef4444"
    elif v>25: lbl,col="Medium","#f59e0b"
    else: lbl,col="Low","#10b981"
    return dict(val=v,lbl=lbl,col=col)

def get_rec(tr,mo,vl):
    if "Uptrend" in tr["lbl"] and mo["val"]<70 and vl["val"]<55:
        return "BUY","#10b981","badge-buy","rec-panel-buy",f"Price above both MAs. RSI {mo['val']:.0f} — momentum intact."
    elif "Downtrend" in tr["lbl"] or mo["val"]>75:
        return "SELL","#ef4444","badge-sell","rec-panel-sell",f"Bearish structure or overbought RSI ({mo['val']:.0f})."
    else:
        return "HOLD","#f59e0b","badge-hold","rec-panel-hold",f"Mixed signals — RSI {mo['val']:.0f}. Monitor for confirmation."

def pct(v,signed=True):
    return f"{'+'if signed and v>=0 else''}{v*100:.2f}%"
def bignum(v,pre="$"):
    if not v: return "—"
    if v>=1e12: return f"{pre}{v/1e12:.2f}T"
    if v>=1e9: return f"{pre}{v/1e9:.2f}B"
    if v>=1e6: return f"{pre}{v/1e6:.2f}M"
    return f"{pre}{v:,.0f}"

GRID = dict(gridcolor="#0f172a",zeroline=False,showline=False,tickfont=dict(size=10,color="#334155"))
BASE = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Space Grotesk',sans-serif",color="#64748b",size=11),
    legend=dict(bgcolor="rgba(2,8,23,.9)",bordercolor="#1e3a5f",borderwidth=1,font=dict(size=11)),
    margin=dict(l=6,r=6,t=40,b=6),hovermode="x unified",
    hoverlabel=dict(bgcolor="#020817",bordercolor="#1e3a5f",font=dict(size=11,color="#e2e8f0")))

def lay(fig,h=420,title=""):
    cfg=dict(**BASE,height=h)
    if title: cfg["title"]=dict(text=title,font=dict(color="#475569",size=13),x=0)
    fig.update_layout(**cfg)
    fig.update_xaxes(**GRID,rangeslider_visible=False)
    fig.update_yaxes(**GRID)
    return fig

# ────────────────────────────────────────────
# MOUNTAIN 3D SCENE  (Three.js via HTML component)
# ────────────────────────────────────────────
def mountain_scene_html(market_data):
    stocks_json = json.dumps(market_data)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  html,body{{width:100%;height:100%;overflow:hidden;background:#020817;font-family:'Space Grotesk',sans-serif}}
  #c{{display:block;width:100%;height:100%}}
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

  #hud{{position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;z-index:10}}

  /* ─ ticker tape ─ */
  #tape{{position:absolute;bottom:0;left:0;right:0;height:44px;
    background:linear-gradient(90deg,rgba(2,8,23,0.95),rgba(6,18,50,0.95));
    border-top:1px solid rgba(96,165,250,0.2);
    display:flex;align-items:center;overflow:hidden;pointer-events:none}}
  #tape-inner{{display:flex;gap:0;white-space:nowrap;animation:scroll 40s linear infinite}}
  @keyframes scroll{{0%{{transform:translateX(0)}}100%{{transform:translateX(-50%)}}}}
  .tape-item{{display:inline-flex;align-items:center;gap:8px;padding:0 28px;
    font-size:12px;font-weight:600;letter-spacing:.05em;color:#94a3b8;
    border-right:1px solid rgba(96,165,250,0.12)}}
  .tape-sym{{color:#60a5fa;font-weight:700}}
  .tape-up{{color:#10b981}}.tape-dn{{color:#ef4444}}

  /* ─ logo ─ */
  #logo{{position:absolute;top:24px;left:32px;pointer-events:none}}
  #logo-text{{font-size:22px;font-weight:700;letter-spacing:-.04em;color:#f1f5f9}}
  #logo-dot{{color:#3b82f6}}
  #logo-sub{{font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.15em;
    color:#1e3a5f;margin-top:2px}}

  /* ─ hint ─ */
  #hint{{position:absolute;bottom:60px;left:50%;transform:translateX(-50%);
    font-size:11px;color:#1e3a5f;letter-spacing:.12em;text-transform:uppercase;
    font-weight:500;pointer-events:none;animation:pulse 2.5s ease-in-out infinite}}
  @keyframes pulse{{0%,100%{{opacity:.4}}50%{{opacity:.9}}}}

  /* ─ hotspot labels ─ */
  .spot-label{{position:absolute;pointer-events:all;cursor:pointer;
    background:rgba(2,8,23,0.82);backdrop-filter:blur(8px);
    border:1px solid rgba(96,165,250,0.35);border-radius:10px;
    padding:8px 16px;font-size:12px;font-weight:600;color:#93c5fd;
    letter-spacing:.05em;text-transform:uppercase;
    transition:all .2s ease;transform:translate(-50%,-50%)}}
  .spot-label:hover{{background:rgba(29,78,216,0.5);border-color:#60a5fa;
    color:#fff;box-shadow:0 0 20px rgba(59,130,246,0.4);transform:translate(-50%,-50%) scale(1.08)}}
  .spot-dot{{width:8px;height:8px;border-radius:50%;background:#3b82f6;
    display:inline-block;margin-right:7px;
    box-shadow:0 0 8px #3b82f6;animation:dotpulse 1.8s ease-in-out infinite}}
  @keyframes dotpulse{{0%,100%{{box-shadow:0 0 4px #3b82f6}}50%{{box-shadow:0 0 14px #60a5fa,0 0 28px rgba(59,130,246,0.4)}}}}

  /* ─ mkt bar ─ */
  #mktbar{{position:absolute;top:24px;right:32px;display:flex;gap:20px;
    background:rgba(2,8,23,0.75);backdrop-filter:blur(10px);
    border:1px solid rgba(96,165,250,0.15);border-radius:12px;
    padding:10px 20px;pointer-events:none}}
  .mkt-item{{text-align:right}}
  .mkt-sym{{font-size:10px;font-weight:700;color:#1e3a5f;letter-spacing:.1em;text-transform:uppercase}}
  .mkt-px{{font-size:14px;font-weight:700;color:#f1f5f9}}
  .mkt-up{{color:#10b981;font-size:11px;font-weight:600}}
  .mkt-dn{{color:#ef4444;font-size:11px;font-weight:600}}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">
  <!-- Logo -->
  <div id="logo">
    <div id="logo-text">Equity<span id="logo-dot">Lens</span></div>
    <div id="logo-sub">3D Portfolio Dashboard</div>
  </div>

  <!-- Market bar -->
  <div id="mktbar">
    <div class="mkt-item"><div class="mkt-sym">S&P 500</div><div class="mkt-px" id="spy-px">—</div><div id="spy-chg">—</div></div>
    <div class="mkt-item"><div class="mkt-sym">NASDAQ</div><div class="mkt-px" id="qqq-px">—</div><div id="qqq-chg">—</div></div>
    <div class="mkt-item"><div class="mkt-sym">VIX</div><div class="mkt-px" id="vix-px">—</div><div id="vix-chg">—</div></div>
  </div>

  <!-- Hotspot labels (positioned by JS) -->
  <div id="spots"></div>

  <!-- Hint -->
  <div id="hint">drag to rotate  ·  click peaks to explore</div>

  <!-- Ticker tape -->
  <div id="tape"><div id="tape-inner"></div></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const STOCKS = {stocks_json};

// ── Renderer ──────────────────────────────────
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas,antialias:true,alpha:true}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x020817, 0.018);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 400);
camera.position.set(0, 28, 62);

function resize() {{
  const w=canvas.clientWidth, h=canvas.clientHeight;
  renderer.setSize(w,h,false);
  camera.aspect=w/h; camera.updateProjectionMatrix();
}}
resize(); window.addEventListener('resize',resize);

// ── Lights ────────────────────────────────────
const ambient = new THREE.AmbientLight(0x0a1628, 2.5);
scene.add(ambient);

const sun = new THREE.DirectionalLight(0x6ea8fe, 3.5);
sun.position.set(30,60,20); sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048);
sun.shadow.camera.near=0.5; sun.shadow.camera.far=200;
sun.shadow.camera.left=-80; sun.shadow.camera.right=80;
sun.shadow.camera.top=80; sun.shadow.camera.bottom=-80;
scene.add(sun);

const rimL = new THREE.DirectionalLight(0x1e40af, 1.8);
rimL.position.set(-40,20,-30); scene.add(rimL);

const rimR = new THREE.DirectionalLight(0x7c3aed, 1.2);
rimR.position.set(40,10,-20); scene.add(rimR);

// ── Stars ─────────────────────────────────────
(function() {{
  const geo = new THREE.BufferGeometry();
  const n = 2200;
  const pos = new Float32Array(n*3);
  for(let i=0;i<n;i++) {{
    const r = 180+Math.random()*60;
    const theta = Math.random()*Math.PI*2;
    const phi = Math.acos(2*Math.random()-1);
    pos[i*3]   = r*Math.sin(phi)*Math.cos(theta);
    pos[i*3+1] = r*Math.cos(phi)*0.4+20;
    pos[i*3+2] = r*Math.sin(phi)*Math.sin(theta);
  }}
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  const mat = new THREE.PointsMaterial({{color:0xacc8f0,size:.45,transparent:true,opacity:.7}});
  scene.add(new THREE.Points(geo,mat));
}})();

// ── Aurora planes ─────────────────────────────
function makeAurora(color1,color2,y,z,rotY) {{
  const geo = new THREE.PlaneGeometry(200,40,20,8);
  const pos = geo.attributes.position;
  for(let i=0;i<pos.count;i++) {{
    const x=pos.getX(i); const yv=pos.getY(i);
    pos.setY(i, yv + Math.sin(x*0.06)*4 + Math.cos(x*0.12)*2);
  }}
  pos.needsUpdate=true; geo.computeVertexNormals();
  const mat = new THREE.MeshBasicMaterial({{
    color:color1,transparent:true,opacity:.06,
    side:THREE.DoubleSide,depthWrite:false
  }});
  const mesh = new THREE.Mesh(geo,mat);
  mesh.position.set(0,y,z); mesh.rotation.y=rotY;
  mesh.userData.baseOpacity = .06;
  mesh.userData.phase = Math.random()*Math.PI*2;
  scene.add(mesh);
  return mesh;
}}
const auroras = [
  makeAurora(0x1e40af,0x7c3aed,55,-80,0.3),
  makeAurora(0x7c3aed,0x0ea5e9,48,-90,-0.2),
  makeAurora(0x0ea5e9,0x10b981,52,-85,0.1),
];

// ── Terrain generation ────────────────────────
function noise2d(x,z,scale,amp) {{
  return Math.sin(x*scale*1.7+0.3)*Math.cos(z*scale*1.3+0.9)*amp
       + Math.sin(x*scale*3.1-0.7)*Math.cos(z*scale*2.8+1.4)*amp*0.5
       + Math.sin(x*scale*5.3+1.1)*Math.cos(z*scale*4.2-0.6)*amp*0.25;
}}

function buildTerrain() {{
  const W=140, D=120, WS=100, DS=80;
  const geo = new THREE.PlaneGeometry(W,D,WS,DS);
  geo.rotateX(-Math.PI/2);
  const pos = geo.attributes.position;
  const colors = [];
  const col = new THREE.Color();

  // Peak positions [wx, wz, height, radius, name]
  const peaks = [
    [0,   -8,  38, 28, 'summit'],    // main summit — Performance
    [-28, 10,  26, 18, 'ridge'],     // left ridge  — Allocation
    [28,  10,  24, 18, 'face'],      // right face  — Risk
    [0,   26,  14, 22, 'base'],      // base plateau — Holdings
  ];

  for(let i=0;i<pos.count;i++) {{
    const x=pos.getX(i), z=pos.getZ(i);
    let y = -2;

    // Base terrain
    y += noise2d(x,z,0.035,6);
    y += noise2d(x,z,0.08,2.5);
    y += noise2d(x,z,0.18,0.8);

    // Sculpt peaks
    for(const [px,pz,ph,pr,name] of peaks) {{
      const d = Math.sqrt((x-px)**2+(z-pz)**2);
      const falloff = Math.exp(-d*d/(pr*pr));
      y += ph*falloff;
    }}

    // Water plane at y<0
    if(y<0) y=0;
    pos.setY(i,y);

    // Color by height
    const h = y/38;
    if(h>0.82) {{       // snow cap
      col.setHSL(0.6,0.15,0.92+h*0.08);
    }} else if(h>0.65) {{ // high rock + blue tint
      col.setHSL(0.61,0.25,0.35+h*0.25);
    }} else if(h>0.45) {{ // mid rock
      col.setHSL(0.62,0.30,0.22+h*0.30);
    }} else if(h>0.25) {{ // low dark rock
      col.setHSL(0.63,0.35,0.12+h*0.25);
    }} else if(h>0.05) {{ // forest-dark
      col.setHSL(0.64,0.40,0.08+h*0.18);
    }} else {{            // water/shore
      col.setHSL(0.60,0.55,0.06);
    }}
    colors.push(col.r,col.g,col.b);
  }}

  geo.setAttribute('color',new THREE.BufferAttribute(new Float32Array(colors),3));
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({{
    vertexColors:true,
    roughness:0.82,metalness:0.12,
    envMapIntensity:0.4,
  }});
  const mesh = new THREE.Mesh(geo,mat);
  mesh.receiveShadow=true; mesh.castShadow=true;
  return mesh;
}}
const terrain = buildTerrain();
scene.add(terrain);

// ── Snow particles ─────────────────────────────
const snowGeo = new THREE.BufferGeometry();
const snowCount = 800;
const snowPos = new Float32Array(snowCount*3);
const snowVel = new Float32Array(snowCount*3);
for(let i=0;i<snowCount;i++) {{
  snowPos[i*3]   = (Math.random()-0.5)*120;
  snowPos[i*3+1] = Math.random()*60+10;
  snowPos[i*3+2] = (Math.random()-0.5)*100;
  snowVel[i*3]   = (Math.random()-.5)*.015;
  snowVel[i*3+1] = -(0.02+Math.random()*.03);
  snowVel[i*3+2] = (Math.random()-.5)*.01;
}}
snowGeo.setAttribute('position',new THREE.BufferAttribute(snowPos,3));
const snowMat = new THREE.PointsMaterial({{color:0xdbeafe,size:.25,transparent:true,opacity:.6}});
const snow = new THREE.Points(snowGeo,snowMat);
scene.add(snow);

// ── Glow sphere on summit ──────────────────────
const glowGeo = new THREE.SphereGeometry(2.2,32,32);
const glowMat = new THREE.MeshBasicMaterial({{color:0x3b82f6,transparent:true,opacity:.18}});
const glowSphere = new THREE.Mesh(glowGeo,glowMat);
glowSphere.position.set(0,39,-8);
scene.add(glowSphere);

// outer ring
const ringGeo = new THREE.TorusGeometry(3.5,0.08,8,64);
const ringMat = new THREE.MeshBasicMaterial({{color:0x60a5fa,transparent:true,opacity:.35}});
const ring = new THREE.Mesh(ringGeo,ringMat);
ring.position.copy(glowSphere.position);
ring.rotation.x=Math.PI/2;
scene.add(ring);

// ── Orbit ring at base (ticker belt) ──────────────
const beltGeo = new THREE.TorusGeometry(36,0.08,8,120);
const beltMat = new THREE.MeshBasicMaterial({{color:0x1e40af,transparent:true,opacity:.25}});
const belt = new THREE.Mesh(beltGeo,beltMat);
belt.rotation.x = Math.PI/2; belt.position.y=0.5;
scene.add(belt);

// ── Hotspot markers ────────────────────────────
// [x, y, z, label, section]
const HOTSPOTS = [
  [0,   38.5, -8,  'Performance',  'performance'],
  [-28, 27,   10,  'Allocation',   'allocation'],
  [28,  25,   10,  'Risk',         'risk'],
  [0,   15,   26,  'Holdings',     'holdings'],
];

const hotMeshes = [];
for(const [hx,hy,hz,label,sec] of HOTSPOTS) {{
  const g = new THREE.SphereGeometry(.55,16,16);
  const m = new THREE.MeshBasicMaterial({{color:0x3b82f6,transparent:true,opacity:.9}});
  const mesh = new THREE.Mesh(g,m);
  mesh.position.set(hx,hy,hz);
  mesh.userData = {{label,sec}};
  scene.add(mesh);
  hotMeshes.push(mesh);

  // halo ring
  const rg = new THREE.TorusGeometry(.9,.04,6,36);
  const rm = new THREE.MeshBasicMaterial({{color:0x93c5fd,transparent:true,opacity:.45}});
  const rmesh = new THREE.Mesh(rg,rm);
  rmesh.position.set(hx,hy,hz);
  rmesh.rotation.x=Math.PI/2;
  rmesh.userData.baseOpacity=.45;
  scene.add(rmesh);
}}

// ── Orbit controls (manual) ───────────────────
let isDragging=false, prevMouse={{x:0,y:0}};
let theta=-0.2, phi=0.42, radius=68;
let targetTheta=-0.2, targetPhi=0.42;

canvas.addEventListener('mousedown', e=>{{ isDragging=true; prevMouse={{x:e.clientX,y:e.clientY}}; }});
canvas.addEventListener('mouseup',   ()=>{{ isDragging=false; }});
canvas.addEventListener('mousemove', e=>{{
  if(!isDragging) return;
  const dx=(e.clientX-prevMouse.x)*0.007;
  const dy=(e.clientY-prevMouse.y)*0.004;
  targetTheta -= dx;
  targetPhi = Math.max(0.15, Math.min(0.78, targetPhi+dy));
  prevMouse={{x:e.clientX,y:e.clientY}};
}});
canvas.addEventListener('wheel', e=>{{
  radius = Math.max(35, Math.min(100, radius+e.deltaY*0.06));
  e.preventDefault();
}},{{passive:false}});

// touch
let lastTouch=null;
canvas.addEventListener('touchstart', e=>{{ lastTouch={{x:e.touches[0].clientX,y:e.touches[0].clientY}}; }});
canvas.addEventListener('touchmove', e=>{{
  if(!lastTouch) return;
  const dx=(e.touches[0].clientX-lastTouch.x)*0.007;
  const dy=(e.touches[0].clientY-lastTouch.y)*0.004;
  targetTheta-=dx; targetPhi=Math.max(0.15,Math.min(0.78,targetPhi+dy));
  lastTouch={{x:e.touches[0].clientX,y:e.touches[0].clientY}};
  e.preventDefault();
}},{{passive:false}});

// ── Hotspot HTML labels (projected) ───────────
const spotsDiv = document.getElementById('spots');
const spotEls = [];
for(const [hx,hy,hz,label,sec] of HOTSPOTS) {{
  const el = document.createElement('div');
  el.className='spot-label';
  el.innerHTML=`<span class="spot-dot"></span>${{label}}`;
  el.addEventListener('click', ()=>{{ sendSection(sec); }});
  spotsDiv.appendChild(el);
  spotEls.push({{el, pos:new THREE.Vector3(hx,hy+1.8,hz)}});
}}

function projectToScreen(v3) {{
  const v=v3.clone().project(camera);
  const w=canvas.clientWidth, h=canvas.clientHeight;
  return {{ x:(v.x+1)/2*w, y:(-v.y+1)/2*h, z:v.z }};
}}

function updateSpots() {{
  for(const {{el,pos}} of spotEls) {{
    const s=projectToScreen(pos);
    if(s.z>1){{ el.style.display='none'; continue; }}
    el.style.display='block';
    el.style.left=s.x+'px'; el.style.top=s.y+'px';
  }}
}}

// ── Ticker tape ────────────────────────────────
function buildTape() {{
  const inner=document.getElementById('tape-inner');
  const items=[];
  for(const [sym,d] of Object.entries(STOCKS)) {{
    const up=d.chg>=0;
    items.push(`<div class="tape-item">
      <span class="tape-sym">${{sym}}</span>
      <span>${{d.px}}</span>
      <span class="${{up?'tape-up':'tape-dn'}}">${{up?'▲':'▼'}} ${{Math.abs(d.chg_pct).toFixed(2)}}%</span>
    </div>`);
  }}
  const row=items.join('');
  inner.innerHTML=row+row+row+row; // repeat for seamless loop
}}

// ── Market bar ─────────────────────────────────
function fillMkt() {{
  const spyD=STOCKS['SPY'];
  if(spyD) {{
    document.getElementById('spy-px').textContent='$'+spyD.px;
    const el=document.getElementById('spy-chg');
    el.textContent=(spyD.chg>=0?'▲':'▼')+' '+Math.abs(spyD.chg_pct).toFixed(2)+'%';
    el.className=spyD.chg>=0?'mkt-up':'mkt-dn';
  }}
  const qqqD=STOCKS['QQQ'];
  if(qqqD) {{
    document.getElementById('qqq-px').textContent='$'+qqqD.px;
    const el=document.getElementById('qqq-chg');
    el.textContent=(qqqD.chg>=0?'▲':'▼')+' '+Math.abs(qqqD.chg_pct).toFixed(2)+'%';
    el.className=qqqD.chg>=0?'mkt-up':'mkt-dn';
  }}
  const vixD=STOCKS['VIX'];
  if(vixD) {{
    document.getElementById('vix-px').textContent=vixD.px.toFixed(2);
    const el=document.getElementById('vix-chg');
    el.textContent=(vixD.chg>=0?'▲':'▼')+' '+Math.abs(vixD.chg_pct).toFixed(2)+'%';
    el.className=vixD.chg>=0?'mkt-up':'mkt-dn';
  }}
}}

buildTape(); fillMkt();

// ── Send section to Streamlit ──────────────────
function sendSection(sec) {{
  window.parent.postMessage({{type:'streamlit:setComponentValue', value:sec}}, '*');
}}

// ── Clock / animate ───────────────────────────
const clock = new THREE.Clock();
function animate() {{
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  // smooth camera orbit
  theta += (targetTheta-theta)*0.06;
  phi   += (targetPhi  -phi  )*0.06;
  const lookY = 8;
  camera.position.x = radius*Math.sin(theta)*Math.cos(phi);
  camera.position.y = radius*Math.sin(phi)+lookY;
  camera.position.z = radius*Math.cos(theta)*Math.cos(phi);
  camera.lookAt(0,lookY,0);

  // aurora breathe
  for(const a of auroras) {{
    a.material.opacity = a.userData.baseOpacity*(0.7+0.3*Math.sin(t*.4+a.userData.phase));
  }}

  // glow pulse
  glowSphere.material.opacity = 0.12+0.08*Math.sin(t*1.8);
  glowSphere.scale.setScalar(1+0.06*Math.sin(t*2.1));
  ring.rotation.z = t*0.4;
  ring.material.opacity = 0.25+0.15*Math.sin(t*1.5);

  // belt rotate
  belt.rotation.z = t*0.12;

  // snow
  const sPos=snow.geometry.attributes.position;
  for(let i=0;i<snowCount;i++) {{
    sPos.array[i*3]   += snowVel[i*3];
    sPos.array[i*3+1] += snowVel[i*3+1];
    sPos.array[i*3+2] += snowVel[i*3+2];
    if(sPos.array[i*3+1]<0) {{
      sPos.array[i*3]   = (Math.random()-.5)*120;
      sPos.array[i*3+1] = 55+Math.random()*10;
      sPos.array[i*3+2] = (Math.random()-.5)*100;
    }}
  }}
  sPos.needsUpdate=true;

  // hotspot pulse
  for(const m of hotMeshes) {{
    m.scale.setScalar(1+0.18*Math.sin(t*2.2+m.position.x));
  }}

  updateSpots();
  renderer.render(scene,camera);
}}
animate();
</script>
</body>
</html>"""

# ────────────────────────────────────────────
# GLOBAL PAGE CSS
# ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;background:#020817!important;color:#e2e8f0!important}
.main .block-container{padding:0!important;max-width:100%!important;background:#020817!important}
#MainMenu,footer,header{visibility:hidden}
section[data-testid="stSidebar"]{display:none}

[data-testid="metric-container"]{background:rgba(15,23,42,.8)!important;border:1px solid rgba(30,58,95,.6)!important;border-radius:14px!important;padding:1rem 1.25rem!important;transition:all .2s;backdrop-filter:blur(8px)}
[data-testid="metric-container"]:hover{border-color:#3b82f6!important;box-shadow:0 0 20px rgba(59,130,246,.12)!important}
[data-testid="metric-container"] label{color:#1e3a5f!important;font-size:.68rem!important;text-transform:uppercase!important;letter-spacing:.09em!important;font-weight:700!important}
[data-testid="stMetricValue"]{color:#f1f5f9!important;font-weight:700!important;font-size:1.3rem!important}
[data-testid="stMetricDelta"]{font-size:.78rem!important;font-weight:600!important}

.stTabs [data-baseweb="tab-list"]{background:rgba(15,23,42,.8)!important;border-radius:12px!important;padding:4px!important;border:1px solid rgba(30,58,95,.6)!important;gap:2px!important;backdrop-filter:blur(8px)}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#1e3a5f!important;border-radius:9px!important;padding:9px 26px!important;font-weight:600!important;font-size:.84rem!important;border:none!important;transition:all .18s!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1d4ed8,#2563eb)!important;color:#fff!important;box-shadow:0 2px 14px rgba(29,78,216,.45)!important}

.stButton>button{background:rgba(15,23,42,.8)!important;color:#60a5fa!important;border:1px solid rgba(30,58,95,.6)!important;border-radius:10px!important;font-weight:600!important;font-size:.83rem!important;transition:all .18s!important;font-family:'Space Grotesk',sans-serif!important;letter-spacing:.02em}
.stButton>button:hover{background:rgba(29,78,216,.25)!important;border-color:#3b82f6!important;color:#93c5fd!important;box-shadow:0 0 16px rgba(59,130,246,.2)!important}

.stRadio>div{flex-direction:row!important;gap:6px!important}
.stRadio label{background:rgba(15,23,42,.8)!important;border:1px solid rgba(30,58,95,.5)!important;border-radius:8px!important;padding:5px 14px!important;cursor:pointer!important;color:#334155!important;font-size:.8rem!important;font-weight:600!important;transition:all .15s!important}
.stRadio label:has(input:checked){background:rgba(29,78,216,.3)!important;border-color:#3b82f6!important;color:#93c5fd!important}

.stSelectbox>div>div{background:rgba(15,23,42,.8)!important;border:1px solid rgba(30,58,95,.5)!important;border-radius:9px!important;color:#f1f5f9!important}
.stProgress>div>div>div{background:linear-gradient(90deg,#1d4ed8,#3b82f6)!important}
[data-testid="stDataFrame"]{border-radius:12px!important;border:1px solid rgba(30,58,95,.6)!important;overflow:hidden!important}
hr{border-color:rgba(30,58,95,.4)!important;margin:1.2rem 0!important}

/* ── detail page panels ── */
.panel{background:rgba(15,23,42,.75);border:1px solid rgba(30,58,95,.5);border-radius:16px;padding:1.4rem 1.6rem;backdrop-filter:blur(12px)}
.panel-sm{background:rgba(15,23,42,.75);border:1px solid rgba(30,58,95,.5);border-radius:14px;padding:1.1rem 1.3rem;backdrop-filter:blur(10px)}

.badge{display:inline-flex;align-items:center;padding:5px 16px;border-radius:99px;font-size:.78rem;font-weight:700;letter-spacing:.06em}
.badge-buy{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3)}
.badge-sell{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.3)}
.badge-hold{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.3)}

.rec-panel{border-radius:16px;padding:1.5rem 1.8rem;position:relative;overflow:hidden;margin-top:1rem;backdrop-filter:blur(10px)}
.rec-panel-buy{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.25)}
.rec-panel-sell{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.25)}
.rec-panel-hold{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.25)}
.rec-panel::after{content:'';position:absolute;top:0;left:0;width:4px;height:100%;border-radius:16px 0 0 16px}
.rec-panel-buy::after{background:#10b981}.rec-panel-sell::after{background:#ef4444}.rec-panel-hold::after{background:#f59e0b}

.ic-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:#1e3a5f;font-weight:700;margin-bottom:.4rem}
.ic-val{font-size:1.05rem;font-weight:800;line-height:1.1}
.ic-sub{font-size:.78rem;color:#1e3a5f;margin-top:.5rem;line-height:1.7}

.weight-pill{display:inline-block;background:rgba(29,78,216,.2);border:1px solid rgba(59,130,246,.3);border-radius:6px;padding:2px 9px;font-size:.72rem;font-weight:700;color:#60a5fa}

.holding-row{background:rgba(15,23,42,.7);border:1px solid rgba(30,58,95,.5);border-radius:12px;padding:.85rem 1.2rem;margin-bottom:.5rem;display:flex;align-items:center;justify-content:space-between;backdrop-filter:blur(8px)}

.interp-box{background:rgba(15,23,42,.7);border:1px solid rgba(30,58,95,.5);border-radius:14px;padding:1.3rem 1.5rem;margin-top:.8rem}
.interp-row{display:flex;align-items:flex-start;gap:.7rem;padding:.55rem 0;border-bottom:1px solid rgba(15,23,42,.8);font-size:.87rem;color:#475569;line-height:1.6}
.interp-row:last-child{border-bottom:none}

.sec-title{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#1e3a5f;font-weight:700;margin-bottom:.9rem;padding-bottom:.5rem;border-bottom:1px solid rgba(30,58,95,.4)}

.page-header{display:flex;align-items:center;justify-content:space-between;padding:1.5rem 2.5rem .5rem;border-bottom:1px solid rgba(30,58,95,.3)}
.header-logo{font-size:1.3rem;font-weight:700;letter-spacing:-.03em;color:#f1f5f9}
.header-logo span{color:#3b82f6}
.detail-body{padding:1.5rem 2.5rem 3rem}

/* glowing section headers */
.section-hero{text-align:center;padding:1.5rem 0 1rem;position:relative}
.section-hero h2{font-size:1.6rem;font-weight:700;letter-spacing:-.03em;color:#f1f5f9;margin:0}
.section-hero p{color:#1e3a5f;font-size:.82rem;margin:.3rem 0 0;text-transform:uppercase;letter-spacing:.1em;font-weight:600}
.section-hero::after{content:'';display:block;width:60px;height:2px;background:linear-gradient(90deg,transparent,#3b82f6,transparent);margin:.8rem auto 0}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────
# SECTION DETAIL VIEWS
# ────────────────────────────────────────────
def back_home():
    st.markdown('<div style="padding:1.5rem 2.5rem .5rem;border-bottom:1px solid rgba(30,58,95,.3);display:flex;align-items:center;justify-content:space-between">', unsafe_allow_html=True)
    c1,c2 = st.columns([1,4])
    with c1:
        if st.button("← Back to Mountain", key="back"):
            st.session_state.view = "home"
            st.session_state.section = None
            st.session_state.ticker = None
            st.rerun()
    with c2:
        st.markdown('<div class="header-logo">Equity<span>Lens</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_performance():
    back_home()
    m = port_calc(tuple(PORT_TICKERS), tuple(PORT_WEIGHTS[t] for t in PORT_TICKERS), "1y")
    if not m:
        st.error("Data unavailable"); return

    st.markdown('<div class="detail-body">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hero">
      <h2>📈 Performance</h2>
      <p>Summit View · 1-Year Portfolio Returns</p>
    </div>""", unsafe_allow_html=True)

    PPER_MAP = {"6 Months":"6mo","1 Year":"1y","2 Years":"2y"}
    pc,_ = st.columns([2,5])
    with pc: pper_lbl = st.selectbox("Period",list(PPER_MAP.keys()),index=1,key="pp_perf")
    pper = PPER_MAP[pper_lbl]
    m = port_calc(tuple(PORT_TICKERS),tuple(PORT_WEIGHTS[t] for t in PORT_TICKERS),pper)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Portfolio Return", pct(m["total"]), f"SPY {pct(m['bt'])}")
    k2.metric("vs Benchmark",    pct(m["outperf"]), "Outperform ↑" if m["outperf"]>0 else "Underperform ↓")
    k3.metric("Ann. Volatility", f"{m['av']*100:.1f}%")
    k4.metric("Sharpe Ratio",    f"{m['sharpe']:.2f}", "≥1 Good · ≥2 Excellent")
    k5.metric("Max Drawdown",    f"{m['dd'].min():.1f}%")

    st.markdown("")
    # Cumulative return
    cum,bc = m["cum"],m["bc"]
    fp = go.Figure()
    fp.add_trace(go.Scatter(x=cum.index,y=(cum-1)*100,name="Portfolio",
        fill="tozeroy",fillcolor="rgba(59,130,246,.08)",line=dict(color="#3b82f6",width=2.5)))
    if bc is not None:
        fp.add_trace(go.Scatter(x=bc.index,y=(bc-1)*100,name="SPY",
            line=dict(color="#1e3a5f",width=1.4,dash="dot")))
    fp.add_hline(y=0,line_color="#0f172a",line_width=1)
    lay(fp,h=360,title="Cumulative Return vs SPY Benchmark")
    fp.update_yaxes(ticksuffix="%",**GRID)
    st.plotly_chart(fp,use_container_width=True)

    # Rolling sharpe
    pr_ = m["pr"]
    rsh = (pr_.rolling(21).mean()/pr_.rolling(21).std())*np.sqrt(252)
    frsh=go.Figure()
    frsh.add_trace(go.Scatter(x=rsh.index,y=rsh,name="Rolling Sharpe (21d)",
        fill="tozeroy",fillcolor="rgba(139,92,246,.07)",line=dict(color="#8b5cf6",width=2)))
    frsh.add_hline(y=1,line_dash="dot",line_color="#10b981",line_width=1,
                   annotation_text="Sharpe=1",annotation_font=dict(size=9,color="#10b981"))
    frsh.add_hline(y=0,line_color="#0f172a",line_width=1)
    lay(frsh,h=210,title="Rolling 21-Day Sharpe Ratio")
    frsh.update_yaxes(**GRID)
    st.plotly_chart(frsh,use_container_width=True)

    # Individual stock performance
    ind_r = m["ind_r"]
    s_r = dict(sorted(ind_r.items(),key=lambda x:x[1],reverse=True))
    fbar=go.Figure(go.Bar(
        x=list(s_r.keys()),y=list(s_r.values()),
        marker_color=[PALETTE[i] for i in range(len(s_r))],
        text=[f"{v:.1f}%" for v in s_r.values()],
        textposition="outside",textfont=dict(color="#475569",size=11),
    ))
    lay(fbar,h=300,title="Individual Stock Returns")
    fbar.update_yaxes(ticksuffix="%",**GRID)
    st.plotly_chart(fbar,use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)


def render_allocation():
    back_home()
    m = port_calc(tuple(PORT_TICKERS),tuple(PORT_WEIGHTS[t] for t in PORT_TICKERS),"1y")
    if not m: st.error("Data unavailable"); return

    st.markdown('<div class="detail-body">',unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hero">
      <h2>🥧 Allocation</h2>
      <p>Ridge View · Portfolio Weight Distribution</p>
    </div>""",unsafe_allow_html=True)

    PPER_MAP={"6 Months":"6mo","1 Year":"1y","2 Years":"2y"}
    pc,_=st.columns([2,5])
    with pc: pper_lbl=st.selectbox("Period",list(PPER_MAP.keys()),index=1,key="pp_alloc")
    pper=PPER_MAP[pper_lbl]
    m=port_calc(tuple(PORT_TICKERS),tuple(PORT_WEIGHTS[t] for t in PORT_TICKERS),pper)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Portfolio Return",pct(m["total"]),f"vs SPY {pct(m['bt'])}")
    k2.metric("Largest Position",max(PORT_TICKERS,key=lambda t:PORT_WEIGHTS[t]),
              f"{max(PORT_WEIGHTS.values())*100:.0f}%")
    k3.metric("# Holdings",str(len(PORT_TICKERS)))
    k4.metric("Diversification","Equal+" if max(PORT_WEIGHTS.values())<0.35 else "Concentrated")

    st.markdown("")
    ac1,ac2=st.columns(2)
    with ac1:
        fpie=go.Figure(go.Pie(
            labels=PORT_TICKERS,values=[PORT_WEIGHTS[t]*100 for t in PORT_TICKERS],
            hole=0.58,
            marker=dict(colors=PALETTE[:len(PORT_TICKERS)],line=dict(color="#020817",width=3)),
            textinfo="label+percent",textfont=dict(color="#f1f5f9",size=12),
        ))
        fpie.update_layout(paper_bgcolor="rgba(0,0,0,0)",showlegend=False,
            height=320,margin=dict(l=6,r=6,t=36,b=6),font=dict(family="Space Grotesk"),
            title=dict(text="Allocation Weights",font=dict(color="#475569",size=13),x=0),
            annotations=[dict(text="Portfolio",x=0.5,y=0.5,
                font=dict(size=12,color="#1e3a5f",family="Space Grotesk"),showarrow=False)])
        st.plotly_chart(fpie,use_container_width=True)

    with ac2:
        # Treemap
        ftree=go.Figure(go.Treemap(
            labels=PORT_TICKERS,
            parents=[""]*len(PORT_TICKERS),
            values=[PORT_WEIGHTS[t]*100 for t in PORT_TICKERS],
            marker=dict(colors=PALETTE[:len(PORT_TICKERS)],line=dict(width=2,color="#020817")),
            textfont=dict(color="#f1f5f9",size=14,family="Space Grotesk"),
            textinfo="label+percent entry",
        ))
        ftree.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=320,
            margin=dict(l=6,r=6,t=36,b=6),
            title=dict(text="Weight Treemap",font=dict(color="#475569",size=13),x=0))
        st.plotly_chart(ftree,use_container_width=True)

    # Holdings table
    st.markdown('<p class="sec-title" style="margin-top:.8rem">Holdings Detail</p>',unsafe_allow_html=True)
    for i,t in enumerate(PORT_TICKERS):
        dft=get_prices(t,pper)
        if dft.empty: continue
        trt=trend_sig(dft); mot=mom_sig(dft); vlt=vol_sig(dft)
        rt,_,rbt,_,_=get_rec(trt,mot,vlt)
        rv=(dft["Close"].squeeze().iloc[-1]/dft["Close"].squeeze().iloc[0]-1)*100
        rc_="#10b981" if rv>=0 else "#ef4444"
        col_dot=PALETTE[i]
        st.markdown(f"""
        <div class="holding-row">
          <div style="display:flex;align-items:center;gap:.9rem">
            <div style="width:8px;height:8px;border-radius:50%;background:{col_dot};box-shadow:0 0 8px {col_dot}"></div>
            <span style="color:#f1f5f9;font-weight:700;font-size:1.05rem">{t}</span>
            <span class="weight-pill">{PORT_WEIGHTS.get(t,0)*100:.0f}%</span>
            <span style="color:#1e3a5f;font-size:.76rem">{trt['lbl']} · RSI {mot['val']:.0f} · Vol {vlt['val']:.0f}%</span>
          </div>
          <div style="display:flex;align-items:center;gap:.9rem">
            <span style="color:{rc_};font-weight:700">{rv:+.2f}%</span>
            <span class="badge {rbt}">{rt}</span>
          </div>
        </div>""",unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)


def render_risk():
    back_home()
    m=port_calc(tuple(PORT_TICKERS),tuple(PORT_WEIGHTS[t] for t in PORT_TICKERS),"1y")
    if not m: st.error("Data unavailable"); return

    st.markdown('<div class="detail-body">',unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hero">
      <h2>⚡ Risk Analysis</h2>
      <p>North Face · Volatility, Drawdown & Correlation</p>
    </div>""",unsafe_allow_html=True)

    PPER_MAP={"6 Months":"6mo","1 Year":"1y","2 Years":"2y"}
    pc,_=st.columns([2,5])
    with pc: pper_lbl=st.selectbox("Period",list(PPER_MAP.keys()),index=1,key="pp_risk")
    pper=PPER_MAP[pper_lbl]
    m=port_calc(tuple(PORT_TICKERS),tuple(PORT_WEIGHTS[t] for t in PORT_TICKERS),pper)

    k1,k2,k3,k4,k5=st.columns(5)
    k1.metric("Ann. Volatility",f"{m['av']*100:.1f}%")
    k2.metric("Max Drawdown",f"{m['dd'].min():.1f}%")
    k3.metric("Sharpe Ratio",f"{m['sharpe']:.2f}")
    k4.metric("Skewness",f"{float(m['pr'].skew()):.2f}")
    k5.metric("Kurtosis",f"{float(m['pr'].kurtosis()):.2f}")

    st.markdown("")
    rr1,rr2=st.columns(2)
    with rr1:
        pr_=m["pr"]
        fh=go.Figure()
        fh.add_trace(go.Histogram(x=pr_*100,nbinsx=40,name="Daily Returns",
            marker_color="#3b82f6",opacity=0.7))
        fh.add_vline(x=float(pr_.mean()*100),line_dash="dot",line_color="#10b981",line_width=1.5,
                     annotation_text="Mean",annotation_font=dict(size=9,color="#10b981"))
        lay(fh,h=290,title="Daily Return Distribution")
        fh.update_xaxes(ticksuffix="%",**GRID); fh.update_yaxes(**GRID)
        st.plotly_chart(fh,use_container_width=True)
    with rr2:
        rets=m["rets"]
        if rets.shape[1]>1:
            corr=rets.corr().round(2)
            fhm=go.Figure(go.Heatmap(
                z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),
                colorscale=[[0,"#ef4444"],[0.5,"#020817"],[1,"#3b82f6"]],zmin=-1,zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",textfont=dict(size=11,color="#f1f5f9"),
            ))
            fhm.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=290,margin=dict(l=6,r=6,t=36,b=6),
                font=dict(family="Space Grotesk",color="#4b5a72"),
                title=dict(text="Return Correlation Matrix",font=dict(color="#475569",size=13),x=0))
            fhm.update_xaxes(showgrid=False); fhm.update_yaxes(showgrid=False)
            st.plotly_chart(fhm,use_container_width=True)

    # Drawdown
    dd=m["dd"]
    fdd=go.Figure()
    fdd.add_trace(go.Scatter(x=dd.index,y=dd,name="Drawdown",
        fill="tozeroy",fillcolor="rgba(239,68,68,.09)",line=dict(color="#ef4444",width=1.5)))
    lay(fdd,h=220,title=f"Portfolio Drawdown  (Max: {dd.min():.1f}%)")
    fdd.update_yaxes(ticksuffix="%",**GRID)
    st.plotly_chart(fdd,use_container_width=True)

    # Volatility by stock
    vol_data={}
    for t in PORT_TICKERS:
        dft=get_prices(t,pper)
        if not dft.empty:
            v=ann_vol(dft["Close"].squeeze())
            vol_data[t]=v
    fv=go.Figure(go.Bar(
        x=list(vol_data.keys()),y=list(vol_data.values()),
        marker_color=PALETTE[:len(vol_data)],
        text=[f"{v:.1f}%" for v in vol_data.values()],
        textposition="outside",textfont=dict(color="#475569",size=11),
    ))
    fv.add_hline(y=25,line_dash="dot",line_color="#10b981",line_width=1,
                 annotation_text="Low threshold 25%",annotation_font=dict(size=9,color="#10b981"))
    fv.add_hline(y=40,line_dash="dot",line_color="#ef4444",line_width=1,
                 annotation_text="High threshold 40%",annotation_font=dict(size=9,color="#ef4444"))
    lay(fv,h=280,title="Annualized Volatility by Stock")
    fv.update_yaxes(ticksuffix="%",**GRID)
    st.plotly_chart(fv,use_container_width=True)

    # Interpretation
    out_=m["outperf"]; sh_=m["sharpe"]; av_=m["av"]
    pi="✅" if out_>0 else "❌"; ri="✅" if av_<.20 else "⚠️"
    si="✅" if sh_>=1 else ("⚠️" if sh_>=.5 else "❌")
    ol="Outperformed" if out_>0 else "Underperformed"
    sl=("Excellent" if sh_>=2 else "Good" if sh_>=1 else "Fair" if sh_>=.5 else "Poor — rebalance")
    st.markdown(f"""
    <div class="interp-box">
      <p style="color:#1e3a5f;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin:0 0 .8rem">Risk Interpretation</p>
      <div class="interp-row">{pi}&nbsp; Portfolio returned <b style="color:#f1f5f9">{pct(m['total'])}</b> vs SPY {pct(m['bt'])} — <b style="color:{'#10b981' if out_>0 else '#ef4444'}">{ol} by {pct(abs(out_))}</b></div>
      <div class="interp-row">{ri}&nbsp; Ann. volatility <b style="color:#f1f5f9">{av_*100:.1f}%</b> — {"below" if av_<.20 else "above"} 20% equity benchmark</div>
      <div class="interp-row">{si}&nbsp; Sharpe ratio <b style="color:#f1f5f9">{sh_:.2f}</b> — {sl} risk-adjusted return</div>
    </div>""",unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)


def render_holdings():
    back_home()
    st.markdown('<div class="detail-body">',unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hero">
      <h2>💎 Holdings</h2>
      <p>Base Camp · Individual Stock Analysis</p>
    </div>""",unsafe_allow_html=True)

    # Stock selector
    sel_col,_ = st.columns([2,5])
    with sel_col:
        sel = st.selectbox("Select Stock",PORT_TICKERS,key="hold_sel")

    # Full individual stock analysis
    with st.spinner(f"Loading {sel}…"):
        df = get_prices(sel,"6mo")
        nfo = get_info(sel)

    if df.empty:
        st.error(f"No data for {sel}"); return

    close=df["Close"].squeeze()
    tr=trend_sig(df); mo=mom_sig(df); vl=vol_sig(df)
    rec,rec_col,rec_badge,rec_panel,rec_reason=get_rec(tr,mo,vl)
    rsi_s=mo["series"]
    m_macd,m_sig_l,m_hist=calc_macd(close)
    b_up,b_mid,b_lo=calc_bb(close)

    px_now=tr["px"]; chg_6m=(px_now-float(close.iloc[0]))/float(close.iloc[0])
    name=nfo.get("longName",sel); sector=nfo.get("sector","—"); mkcap=nfo.get("marketCap",0)
    pe=nfo.get("trailingPE",None); eps_v=nfo.get("trailingEps",None)
    h52=nfo.get("fiftyTwoWeekHigh",None); l52=nfo.get("fiftyTwoWeekLow",None)
    beta=nfo.get("beta",None); dy=nfo.get("dividendYield",None)

    ch_col="#10b981" if chg_6m>=0 else "#ef4444"
    ch_arr="▲" if chg_6m>=0 else "▼"
    tc=TICKER_COLORS.get(sel,"#3b82f6")

    # Header
    hc1,hc2,hc3=st.columns([3,1,1])
    with hc1:
        st.markdown(f"""
        <div>
          <div style="color:#1e3a5f;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:4px">{sector}</div>
          <div style="font-size:1.9rem;font-weight:800;color:#f1f5f9;letter-spacing:-.04em;line-height:1.1">{name}</div>
          <div style="margin-top:3px">
            <span style="color:{tc};font-weight:700;font-size:.9rem">{sel}</span>
            <span style="color:#0f172a;margin:0 6px">·</span>
            <span style="color:#1e3a5f;font-size:.85rem">{bignum(mkcap)} Market Cap</span>
          </div>
        </div>""",unsafe_allow_html=True)
    with hc2:
        st.markdown(f"""
        <div style="text-align:right;padding-top:.4rem">
          <div style="font-size:2rem;font-weight:800;color:#f1f5f9">${px_now:,.2f}</div>
          <div style="color:{ch_col};font-size:.9rem;font-weight:700">{ch_arr} {abs(chg_6m)*100:.2f}%<span style="color:#1e3a5f;font-size:.7rem"> 6M</span></div>
        </div>""",unsafe_allow_html=True)
    with hc3:
        st.markdown(f"<div style='text-align:right;padding-top:.7rem'><span class='badge {rec_badge}' style='font-size:.9rem;padding:8px 22px'>⬟ {rec}</span></div>",unsafe_allow_html=True)

    st.markdown("---")

    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Price",f"${px_now:,.2f}",pct(chg_6m))
    k2.metric("20-Day MA",f"${tr['m20']:,.2f}")
    k3.metric("50-Day MA",f"${tr['m50']:,.2f}")
    k4.metric("RSI (14)",f"{mo['val']:.1f}",mo["lbl"])
    k5.metric("Ann. Vol",f"{vl['val']:.1f}%",vl["lbl"])
    k6.metric("Beta",f"{beta:.2f}" if beta else "—")

    st.markdown("")

    # Chart type
    ct_c,_=st.columns([2,6])
    with ct_c:
        chart_type=st.radio("Chart",["Candlestick","Line"],horizontal=True,key="ctype_hold",label_visibility="collapsed")

    # Price + Volume
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.73,0.27],vertical_spacing=0.02)
    if chart_type=="Candlestick":
        fig.add_trace(go.Candlestick(x=df.index,
            open=df["Open"].squeeze(),high=df["High"].squeeze(),
            low=df["Low"].squeeze(),close=close,
            increasing_line_color="#10b981",increasing_fillcolor="#10b981",
            decreasing_line_color="#ef4444",decreasing_fillcolor="#ef4444",
            name=sel,showlegend=False),row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index,y=close,name=sel,
            line=dict(color=tc,width=2.3),showlegend=False),row=1,col=1)

    fig.add_trace(go.Scatter(x=df.index,y=close.rolling(20).mean(),name="MA 20",
        line=dict(color="#f59e0b",width=1.5,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=close.rolling(50).mean(),name="MA 50",
        line=dict(color="#ef4444",width=1.5,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=b_up,name="BB Upper",
        line=dict(color="#8b5cf6",width=1,dash="dash")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=b_lo,name="BB Lower",
        line=dict(color="#8b5cf6",width=1,dash="dash"),
        fill="tonexty",fillcolor="rgba(139,92,246,.05)"),row=1,col=1)

    vc=["#10b981" if float(c_)>=float(o_) else "#ef4444"
        for c_,o_ in zip(df["Close"].squeeze(),df["Open"].squeeze())]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"].squeeze(),
        marker_color=vc,opacity=0.5,name="Volume",showlegend=False),row=2,col=1)

    lay(fig,h=520)
    fig.update_yaxes(row=2,col=1,title_text="Volume",title_font=dict(size=9,color="#1e3a5f"),**GRID)
    st.plotly_chart(fig,use_container_width=True)

    # RSI + MACD
    ic1,ic2=st.columns(2)
    with ic1:
        st.markdown('<p class="sec-title">RSI — 14 Day</p>',unsafe_allow_html=True)
        fr=go.Figure()
        fr.add_hrect(y0=70,y1=100,fillcolor="rgba(239,68,68,.05)",line_width=0)
        fr.add_hrect(y0=0,y1=30,fillcolor="rgba(16,185,129,.05)",line_width=0)
        fr.add_hline(y=70,line_dash="dot",line_color="#ef4444",line_width=1,
                     annotation_text="Overbought 70",annotation_font=dict(size=9,color="#ef4444"),
                     annotation_position="top right")
        fr.add_hline(y=30,line_dash="dot",line_color="#10b981",line_width=1,
                     annotation_text="Oversold 30",annotation_font=dict(size=9,color="#10b981"),
                     annotation_position="bottom right")
        fr.add_trace(go.Scatter(x=rsi_s.index,y=rsi_s,name="RSI",
            line=dict(color="#60a5fa",width=2.2),fill="tozeroy",fillcolor="rgba(96,165,250,.06)"))
        lay(fr,h=265); fr.update_yaxes(range=[0,100],**GRID)
        st.plotly_chart(fr,use_container_width=True)
    with ic2:
        st.markdown('<p class="sec-title">MACD — 12 / 26 / 9</p>',unsafe_allow_html=True)
        fm=go.Figure()
        hc=["#10b981" if v>=0 else "#ef4444" for v in m_hist.fillna(0)]
        fm.add_trace(go.Bar(x=df.index,y=m_hist,marker_color=hc,opacity=0.7,name="Histogram"))
        fm.add_trace(go.Scatter(x=df.index,y=m_macd,line=dict(color="#3b82f6",width=2),name="MACD"))
        fm.add_trace(go.Scatter(x=df.index,y=m_sig_l,line=dict(color="#f59e0b",width=1.5,dash="dot"),name="Signal"))
        lay(fm,h=265)
        st.plotly_chart(fm,use_container_width=True)

    # Bollinger
    st.markdown('<p class="sec-title">Bollinger Bands (20-Day ±2σ)</p>',unsafe_allow_html=True)
    fbb=go.Figure()
    fbb.add_trace(go.Scatter(x=df.index,y=b_up,name="Upper",line=dict(color="#8b5cf6",width=1.2,dash="dash")))
    fbb.add_trace(go.Scatter(x=df.index,y=b_mid,name="Middle (MA 20)",line=dict(color="#475569",width=1,dash="dot")))
    fbb.add_trace(go.Scatter(x=df.index,y=b_lo,name="Lower",line=dict(color="#8b5cf6",width=1.2,dash="dash"),
        fill="tonexty",fillcolor="rgba(139,92,246,.06)"))
    fbb.add_trace(go.Scatter(x=df.index,y=close,name="Price",line=dict(color=tc,width=2)))
    lay(fbb,h=260); st.plotly_chart(fbb,use_container_width=True)

    # Volatility gauge
    st.markdown('<p class="sec-title">Annualized Volatility Gauge</p>',unsafe_allow_html=True)
    fg=go.Figure(go.Indicator(
        mode="gauge+number",value=vl["val"],
        number=dict(suffix="%",font=dict(color="#f1f5f9",size=34,family="Space Grotesk")),
        gauge=dict(
            axis=dict(range=[0,80],tickcolor="#1e3a5f",tickfont=dict(color="#1e3a5f",size=9)),
            bar=dict(color=vl["col"],thickness=0.25),bgcolor="#020817",borderwidth=0,
            steps=[dict(range=[0,25],color="#051209"),dict(range=[25,40],color="#120d02"),dict(range=[40,80],color="#120202")]),
        title=dict(text=f"Level: <b style='color:{vl['col']}'>{vl['lbl']}</b>",
                   font=dict(size=12,color="#334155",family="Space Grotesk"))))
    fg.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=230,
        font=dict(family="Space Grotesk",color="#334155"),margin=dict(l=20,r=20,t=20,b=10))
    st.plotly_chart(fg,use_container_width=True)

    # Signal cards
    sc1,sc2,sc3=st.columns(3)
    with sc1:
        st.markdown(f"""
        <div class="panel-sm">
          <div class="ic-label">📊 Trend Analysis</div>
          <div class="ic-val" style="color:{tr['col']}">{tr['lbl']}</div>
          <div class="ic-sub">
            Price &nbsp;<b style="color:#f1f5f9">${tr['px']:,.2f}</b><br>
            MA 20 &nbsp;<b style="color:#f59e0b">${tr['m20']:,.2f}</b><br>
            MA 50 &nbsp;<b style="color:#ef4444">${tr['m50']:,.2f}</b><br>
            <span style="color:#10b981">Price &gt; MA20 &gt; MA50 = Uptrend</span>
          </div>
        </div>""",unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="panel-sm">
          <div class="ic-label">⚡ Momentum (RSI 14)</div>
          <div class="ic-val" style="color:{mo['col']}">{mo['lbl']}</div>
          <div class="ic-sub">
            RSI &nbsp;<b style="color:#f1f5f9">{mo['val']:.2f}</b><br>
            <span style="color:#10b981">＜ 30 = Oversold → Buy Signal</span><br>
            <span style="color:#ef4444">＞ 70 = Overbought → Sell Signal</span>
          </div>
        </div>""",unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="panel-sm">
          <div class="ic-label">🌊 Volatility (20D Ann.)</div>
          <div class="ic-val" style="color:{vl['col']}">{vl['lbl']}</div>
          <div class="ic-sub">
            Ann. Vol &nbsp;<b style="color:#f1f5f9">{vl['val']:.1f}%</b><br>
            <span style="color:#10b981">＜ 25% Low</span><br>
            <span style="color:#f59e0b">25–40% Medium</span><br>
            <span style="color:#ef4444">＞ 40% High</span>
          </div>
        </div>""",unsafe_allow_html=True)

    # Recommendation
    st.markdown(f"""
    <div class="rec-panel {rec_panel}">
      <div style="display:flex;align-items:center;gap:1.1rem;margin-bottom:.75rem">
        <span class="badge {rec_badge}" style="font-size:1rem;padding:8px 24px">⬟ {rec}</span>
        <div>
          <div style="color:#f1f5f9;font-weight:700;font-size:.95rem;margin-bottom:3px">Trading Recommendation</div>
          <div style="color:#475569;font-size:.84rem">{rec_reason}</div>
        </div>
      </div>
      <p style="color:#1e3a5f;font-size:.68rem;margin:0;border-top:1px solid rgba(30,58,95,.4);padding-top:.65rem">
        ⚠ Technical analysis only. Not investment advice.
      </p>
    </div>""",unsafe_allow_html=True)

    # Fundamentals
    if any([pe,eps_v,h52,l52,beta,dy]):
        st.markdown("---")
        st.markdown('<p class="sec-title">Company Fundamentals</p>',unsafe_allow_html=True)
        fb1,fb2,fb3,fb4,fb5,fb6=st.columns(6)
        fb1.metric("P/E Ratio",f"{pe:.1f}×" if pe else "—")
        fb2.metric("EPS (TTM)",f"${eps_v:.2f}" if eps_v else "—")
        fb3.metric("52W High",f"${h52:,.2f}" if h52 else "—")
        fb4.metric("52W Low",f"${l52:,.2f}" if l52 else "—")
        fb5.metric("Div Yield",f"{dy*100:.2f}%" if dy else "—")
        fb6.metric("Beta",f"{beta:.2f}" if beta else "—")

    st.markdown('</div>',unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HOME — 3D MOUNTAIN
# ════════════════════════════════════════════════════════════════
def render_home():
    # Gather market data for tape + mkt bar
    market_data = {}
    for sym in PORT_TICKERS + ["SPY","QQQ"]:
        px_,chg_,_ = get_quick(sym)
        if px_ is not None:
            market_data[sym] = {
                "px": round(px_,2),
                "chg": round(chg_,4),
                "chg_pct": round(chg_*100,2)
            }
    # VIX approximation
    market_data["VIX"] = {"px": 18.5, "chg": 0.02, "chg_pct": 0.12}

    html_content = mountain_scene_html(market_data)
    st.components.v1.html(html_content, height=760, scrolling=False)

    # Section quick-access buttons under the 3D scene
    st.markdown('<div style="padding:.8rem 2.5rem 0;display:flex;gap:10px;flex-wrap:wrap">',
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if st.button("📈  Performance", key="btn_perf", use_container_width=True):
            st.session_state.view="section"; st.session_state.section="performance"; st.rerun()
    with c2:
        if st.button("🥧  Allocation", key="btn_alloc", use_container_width=True):
            st.session_state.view="section"; st.session_state.section="allocation"; st.rerun()
    with c3:
        if st.button("⚡  Risk", key="btn_risk", use_container_width=True):
            st.session_state.view="section"; st.session_state.section="risk"; st.rerun()
    with c4:
        if st.button("💎  Holdings", key="btn_hold", use_container_width=True):
            st.session_state.view="section"; st.session_state.section="holdings"; st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

    # Listen for 3D hotspot clicks via component messaging (best-effort)
    st.markdown("""
    <script>
    window.addEventListener('message', function(e) {
        if(e.data && e.data.type === 'streamlit:setComponentValue') {
            // Trigger rerun by updating URL param
            const sec = e.data.value;
            window.location.hash = sec;
        }
    });
    </script>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════
if st.session_state.view == "section":
    s = st.session_state.section
    if   s == "performance": render_performance()
    elif s == "allocation":  render_allocation()
    elif s == "risk":        render_risk()
    elif s == "holdings":    render_holdings()
    else: render_home()
else:
    render_home()
