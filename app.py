"""Responsive Streamlit virtual sunglasses try-on app."""
import copy, io, base64, json
import cv2, numpy as np, streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from tryon.assets import STYLES, load_glasses
from tryon.landmarks import FaceDetector
from tryon.pipeline import TryOnError, TryOnOptions, load_image_bytes, run_tryon

st.set_page_config(page_title="AuraLens", page_icon="🕶️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{max-width:1180px;padding:1.1rem clamp(.8rem,4vw,3rem) 3rem}h1{letter-spacing:-.04em}[data-testid="stRadio"]>div{gap:.55rem;overflow-x:auto;flex-wrap:nowrap;padding-bottom:.35rem}[data-testid="stRadio"] label{min-width:92px;padding:.55rem .7rem;border:1px solid #e6e8ee;border-radius:14px;background:#fff}@media(max-width:640px){.block-container{padding:.75rem .7rem 2rem}h1{font-size:2rem}[data-testid="stRadio"] label{min-width:82px;font-size:.82rem}}</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading face-landmark model…")
def get_detector(): return FaceDetector()
def to_bytes(rgb, fmt="PNG"):
    buf=io.BytesIO(); Image.fromarray(rgb).save(buf, format=fmt, quality=95 if fmt=="JPEG" else None); return buf.getvalue()
def split_view(before, after, pct):
    x=int(before.shape[1]*pct/100); out=after.copy(); out[:,:x]=before[:,:x]
    if 0<x<before.shape[1]: cv2.line(out,(x,0),(x,before.shape[0]),(255,255,255),max(2,before.shape[1]//400))
    return out
def variants(style):
    base=load_glasses(style); palettes=[(1,1,1),(.82,.88,1.08),(1.12,.86,.72),(.72,1.04,.82),(1.10,1.05,.78),(.90,.76,1.12),(1.18,.92,.64),(.78,.92,1.15),(1.08,.78,.92),(.82,1.08,1.05)]
    out=[]
    for i,mult in enumerate(palettes,1):
        g=copy.copy(base); g.name=f"{style} {i}"; g.rgb=np.clip(g.rgb*np.array(mult,np.float32),0,1)
        # Real geometric variation: each option has a distinct width/height profile.
        g.rgb=np.ascontiguousarray(g.rgb); g.alpha=np.ascontiguousarray(g.alpha)
        g.variant_scale=(0.86 + i * 0.03, 0.90 + ((i * 7) % 5) * 0.035)
        out.append(g)
    return out

def live_camera(glasses_by_style, active_style, live_scale, live_dy, live_opacity, live_wearing, live_thorough):
    payload = {}
    for style, items in glasses_by_style.items():
        payload[style] = []
        for g in items:
            # Keep frame edges solid, but make the lens material translucent so the user's
            # eyes, skin tone, and scene lighting remain visible through the glass.
            lens = np.clip(g.lens_mask, 0, 1)
            frame_alpha = np.clip(g.alpha - lens, 0, 1)
            lens_alpha = g.alpha * (1.0 - 0.34 * lens)
            live_alpha = np.maximum(frame_alpha, lens_alpha)
            rgba = np.dstack([np.clip(g.rgb * 255, 0, 255).astype(np.uint8), np.clip(live_alpha * 255, 0, 255).astype(np.uint8)])
            b = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(b, format="PNG")
            sx, sy = getattr(g, "variant_scale", (1, 1)); payload[style].append({"src": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode(), "sx": sx, "sy": sy})
    data = json.dumps(payload)
    html = f'''<!doctype html><html><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#fff;font-family:Arial;color:white;overflow:hidden}}
    #stage{{position:relative;width:100%;height:360px;background:#111}}video,canvas{{position:absolute;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}}
    #msg{{position:absolute;inset:45% 0 auto;text-align:center;color:#fff;background:#0008;padding:10px}}
    #styles{{position:absolute;bottom:78px;left:0;right:0;display:flex;gap:6px;justify-content:center}}#tray{{position:absolute;bottom:0;left:0;right:0;display:flex;gap:8px;justify-content:center;overflow-x:auto;padding:10px;background:linear-gradient(transparent,#000d)}}#capture{{display:block;width:100%;height:42px;border:0;border-top:1px solid #ddd;border-radius:14px 14px 0 0;background:#fff;color:#222;font-size:14px;cursor:pointer}}
    button{{flex:0 0 66px;height:58px;border:2px solid #fff8;border-radius:14px;background:#fff;color:#222;font-size:12px}}
    button.active{{border-color:#00e0d0;background:#dffffb}}button img{{max-width:58px;max-height:38px}}
    @media(max-width:600px){{#stage{{height:360px}}#capture{{height:42px;font-size:14px}}}}
    </style><div id="stage"><video id="video" autoplay playsinline></video><canvas id="canvas"></canvas><div id="msg">Starting camera…</div><div id="styles"></div><div id="tray"></div></div><button id="capture" type="button">Take photo</button>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script><script>
    const catalog={data}; const fit={{scale:{live_scale}*{1.08 if live_wearing else 1.0},dy:{live_dy},opacity:{max(live_opacity, 0.95) if live_wearing else live_opacity},wearing:{str(live_wearing).lower()},thorough:{str(live_thorough).lower()}}}; let style='{active_style}', idx=0, stream, video=document.getElementById('video'), canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d'), msg=document.getElementById('msg'), tray=document.getElementById('tray'), img=new Image();
    function makeTray(){{tray.innerHTML='';catalog[style].forEach((g,i)=>{{let b=document.createElement('button');b.innerHTML='<img src="'+g.src+'">';b.onclick=()=>{{idx=i;img.src=g.src;document.querySelectorAll('#tray button').forEach(x=>x.classList.remove('active'));b.classList.add('active')}};tray.appendChild(b)}});tray.children[idx]?.classList.add('active');img.src=catalog[style][idx].src}}
    async function start(){{try{{stream=await navigator.mediaDevices.getUserMedia({{video:{{facingMode:'user',width:{{ideal:640}},height:{{ideal:480}},frameRate:{{ideal:24,max:30}}}},audio:false}});video.srcObject=stream;msg.style.display='none';loop()}}catch(e){{msg.textContent='Camera access is unavailable. Allow camera permission or use Upload photo below.'}}}}
    const mesh=new FaceMesh({{locateFile:f=>'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/'+f}});mesh.setOptions({{maxNumFaces:{4 if live_thorough else 1},refineLandmarks:true,minDetectionConfidence:.65,minTrackingConfidence:.65}});
    let smooth=null;
    function avg(p,ids){{let x=0,y=0;ids.forEach(i=>{{x+=p[i].x;y+=p[i].y}});return {{x:x/ids.length,y:y/ids.length}}}}
    mesh.onResults(r=>{{ctx.clearRect(0,0,canvas.width,canvas.height);if(!r.multiFaceLandmarks?.length){{smooth=null;return}}let p=r.multiFaceLandmarks[0], left=avg(p,[33,133]), right=avg(p,[362,263]), cx=(left.x+right.x)/2*canvas.width, cy=(left.y+right.y)/2*canvas.height, dist=Math.hypot((right.x-left.x)*canvas.width,(right.y-left.y)*canvas.height), angle=Math.atan2((right.y-left.y)*canvas.height,(right.x-left.x)*canvas.width),g=catalog[style][idx],w=dist/0.50*g.sx*fit.scale,h=img.naturalHeight/img.naturalWidth*w*g.sy;let target={{x:cx,y:cy+fit.dy*dist,w,h,a:angle}};smooth=smooth||target;smooth.x+=(target.x-smooth.x)*.35;smooth.y+=(target.y-smooth.y)*.35;smooth.w+=(target.w-smooth.w)*.35;smooth.h+=(target.h-smooth.h)*.35;smooth.a+=(target.a-smooth.a)*.35;
    ctx.save();ctx.globalAlpha=.55+.45*fit.opacity;ctx.translate(smooth.x,smooth.y);ctx.rotate(smooth.a);ctx.drawImage(img,-smooth.w/2,-smooth.h*0.50,smooth.w,smooth.h);ctx.restore()}});
    let lastFrame=0;async function loop(now=0){{if(video.readyState>=2 && now-lastFrame>=42){{lastFrame=now;let w=video.videoWidth,h=video.videoHeight,scale=Math.min(1,640/Math.max(w,h));canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));await mesh.send({{image:video}})}}requestAnimationFrame(loop)}}
    document.getElementById('capture').onclick=()=>{{if(!canvas.width)return;let shot=document.createElement('canvas');shot.width=canvas.width;shot.height=canvas.height;let sc=shot.getContext('2d');sc.drawImage(video,0,0,shot.width,shot.height);sc.drawImage(canvas,0,0);let a=document.createElement('a');a.download='sunglasses-try-on.png';a.href=shot.toDataURL('image/png');a.click()}};
    makeTray();start();
    </script></html>'''
    components.html(html, height=410, scrolling=False)

st.title("🕶️ AuraLens")
with st.sidebar:
    st.header("AuraLens")
    style=st.selectbox("Style",STYLES,index=1,format_func=str.capitalize)
    st.header("Fit & realism"); scale=st.slider("Size",.80,1.30,1.0,.01); dy=st.slider("Vertical offset",-.40,.40,0.0,.01); opacity=st.slider("Lens darkness",.40,1.0,.90,.01); match=st.checkbox("Match photo lighting / noise / blur",True); shadow=st.slider("Shadow on face",0.0,.6,.30,.01); wearing=st.checkbox("Already wearing glasses",False); thorough=st.checkbox("Search for small / extra faces",False)
st.subheader("Camera Try-on")
all_variants={s:variants(s) for s in STYLES}
live_camera(all_variants, style, scale, dy, opacity, wearing, thorough)
choices=all_variants[style]
selected=0
uploads=st.file_uploader("Upload your photos",type=["jpg","jpeg","png","webp","bmp"],accept_multiple_files=True,key="photo")
if not uploads: st.stop()
for photo_index, uploaded in enumerate(uploads, 1):
    try:
        rgb=load_image_bytes(uploaded.getvalue(),getattr(uploaded,"name",f"photo_{photo_index}.jpg")); opts=TryOnOptions(style=style,glasses=choices[selected],scale=scale,dy=dy,lens_opacity=opacity,shadow=shadow,match_photo=match,thorough=thorough,wearing_glasses=wearing,draw_arms=False)
        with st.spinner(f"Fitting glasses to photo {photo_index}…"): result=run_tryon(rgb,opts,detector=get_detector())
    except TryOnError as e:
        st.error(f"{getattr(uploaded,'name',f'Photo {photo_index}')}: {e}"); continue
    for warning in result.warnings: st.warning(f"Photo {photo_index}: {warning}")
    st.subheader(f"Photo {photo_index}: {getattr(uploaded,'name', 'uploaded image')}")
    tab1,tab2,tab3=st.tabs(["Before / after","Compare","Original"])
    with tab1:
        c1,c2=st.columns(2); c1.image(result.original,caption="Before",use_container_width=True); c2.image(result.output,caption="After",use_container_width=True)
    with tab2:
        pct=st.slider("Original ← → Glasses",0,100,50,key=f"cmp_{photo_index}"); st.image(split_view(result.original,result.output,pct),use_container_width=True)
    with tab3: st.image(result.original,use_container_width=True)
    c1,c2=st.columns(2); c1.download_button("Download PNG",to_bytes(result.output),f"tryon_result_{photo_index}.png","image/png",use_container_width=True); c2.download_button("Download JPG",to_bytes(result.output,"JPEG"),f"tryon_result_{photo_index}.jpg","image/jpeg",use_container_width=True)
