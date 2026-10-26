const BEST_KEY = 'mufc-penalty-best';
const ROUND_SIZE = 5;
const DT = 1 / 120;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const lerp = (a, b, t) => a + (b - a) * t;
export const PENALTY_PHYSICS = Object.freeze({ goalWidth:7.32, goalHeight:2.44, goalZ:11, netDepth:1.45, ballRadius:.11, postRadius:.06, gravity:9.81, drag:.0055, timeStep:DT });
export const PENALTY_MODES = Object.freeze({
  placement:{name:'推射',baseSpeed:11,addedSpeed:13,dispersion:.045,spinFactor:.22,detail:'落点稳定，球速较低'},
  power:{name:'大力',baseSpeed:17,addedSpeed:19,dispersion:.18,spinFactor:.16,detail:'球速更高，落点偏差更大'},
  curl:{name:'弧线',baseSpeed:13,addedSpeed:14,dispersion:.085,spinFactor:1,detail:'旋转产生更明显的侧向弧线'},
});
function randomGenerator(seed){let state=seed>>>0;return()=>{state+=0x6D2B79F5;let v=Math.imul(state^state>>>15,1|state);v^=v+Math.imul(v^v>>>7,61|v);return((v^v>>>14)>>>0)/4294967296;};}
function gaussian(random){return Math.sqrt(-2*Math.log(Math.max(.000001,random())))*Math.cos(2*Math.PI*random());}
function freshSeed(){try{return crypto.getRandomValues(new Uint32Array(1))[0];}catch{return Math.floor(Math.random()*4294967296);}}
function readBest(){try{const v=Number(localStorage.getItem(BEST_KEY));return Number.isInteger(v)&&v>=0&&v<=ROUND_SIZE?v:0;}catch{return 0;}}
function initialVelocity(target,speed){
  const range=Math.hypot(target.x,PENALTY_PHYSICS.goalZ),height=target.y-PENALTY_PHYSICS.ballRadius,sq=speed*speed;
  const discriminant=sq*sq-PENALTY_PHYSICS.gravity*(PENALTY_PHYSICS.gravity*range*range+2*height*sq);
  const angle=discriminant>0?Math.atan((sq-Math.sqrt(discriminant))/(PENALTY_PHYSICS.gravity*range)):Math.atan2(height+3,range);
  const horizontal=speed*Math.cos(angle);
  return{x:horizontal*target.x/range,y:speed*Math.sin(angle),z:horizontal*PENALTY_PHYSICS.goalZ/range};
}
function pointDistanceToSegment(p,a,b){const dx=b.x-a.x,dy=b.y-a.y,den=dx*dx+dy*dy,t=den?clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/den,0,1):0;return Math.hypot(p.x-a.x-dx*t,p.y-a.y-dy*t);}
function keeperSkeleton(k){
  const up={x:Math.sin(k.angle),y:Math.cos(k.angle)},side={x:Math.cos(k.angle),y:-Math.sin(k.angle)};
  const p=(along,across=0)=>({x:k.x+up.x*along+side.x*across,y:k.y+up.y*along+side.y*across});
  const hip=p(0),shoulder=p(.54),head=p(.8),reach=k.diving?.7:.48,lift=k.diving?.18:.05;
  const leftElbow=p(.4+lift,-.34),rightElbow=p(.4+lift,.34),leftHand=p(.47+lift,-reach),rightHand=p(.47+lift,reach),leftKnee=p(-.29,-.16),rightKnee=p(-.27,.18),leftFoot=p(-.66,-.25),rightFoot=p(-.64,.27);
  const segments=[{a:hip,b:shoulder,radius:.19,part:'body'},{a:shoulder,b:leftElbow,radius:.085,part:'arm'},{a:leftElbow,b:leftHand,radius:.085,part:'arm'},{a:shoulder,b:rightElbow,radius:.085,part:'arm'},{a:rightElbow,b:rightHand,radius:.085,part:'arm'},{a:hip,b:leftKnee,radius:.11,part:'leg'},{a:leftKnee,b:leftFoot,radius:.09,part:'leg'},{a:hip,b:rightKnee,radius:.11,part:'leg'},{a:rightKnee,b:rightFoot,radius:.09,part:'leg'}];
  return{hip,shoulder,head,leftElbow,rightElbow,leftHand,rightHand,leftKnee,rightKnee,leftFoot,rightFoot,segments};
}
function keeperCollision(previous,ball,keeper){
  const plane=PENALTY_PHYSICS.goalZ-.38,t=ball.z===previous.z?1:clamp((plane-previous.z)/(ball.z-previous.z),0,1);
  const p={x:lerp(previous.x,ball.x,t),y:lerp(previous.y,ball.y,t)},depth=Math.abs(lerp(previous.z,ball.z,t)-plane);
  if(depth>.34)return null;
  const skeleton=keeperSkeleton(keeper);
  for(const s of skeleton.segments)if(Math.hypot(pointDistanceToSegment(p,s.a,s.b),depth)<=s.radius+PENALTY_PHYSICS.ballRadius)return{point:{...p,z:plane},part:s.part};
  for(const [joint,radius,part]of[[skeleton.head,.145,'head'],[skeleton.leftHand,.125,'glove'],[skeleton.rightHand,.125,'glove']])if(Math.hypot(p.x-joint.x,p.y-joint.y,depth)<=radius+PENALTY_PHYSICS.ballRadius)return{point:{...p,z:plane},part};
  return null;
}
function updateKeeper(k,ball,time,spin,c){
  if(!c.enabled)return;
  if(!k.target&&time>=k.reaction){
    const t=clamp((PENALTY_PHYSICS.goalZ-.38-ball.z)/Math.max(1,ball.vz),0,1.8);
    // A delayed, imperfect extrapolation from the observed ball. No knowledge
    // of the shooter's target or the eventual categorical result is provided.
    k.target={x:ball.x+ball.vx*t+.5*spin*5.5*t*t+k.errorX,y:ball.y+ball.vy*t-.5*PENALTY_PHYSICS.gravity*t*t+k.errorY};k.diveStart=time+.065;
  }
  if(!k.target)return;
  const direction=Math.sign(k.target.x)||1;k.diving=time>=k.diveStart&&(Math.abs(k.target.x)>.62||k.target.y>1.55);
  const target=clamp(k.target.x-(k.diving?direction*.44:0),-3.25,3.25),maxSpeed=k.diving?c.diveSpeed:c.moveSpeed;
  k.vx=clamp(k.vx+clamp((target-k.x)*28-k.vx*5,-c.acceleration,c.acceleration)*DT,-maxSpeed,maxSpeed);k.x+=k.vx*DT;
  if(k.diving&&!k.jumped){k.jumped=true;k.vy=k.target.y>1.5?c.jumpSpeed:k.target.y>.9?c.jumpSpeed*.65:.5;}
  if(k.jumped){k.vy-=8*DT;k.y=Math.max(.53,k.y+k.vy*DT);}
  const angle=k.diving?direction*Math.min(1.28,.6+Math.abs(k.target.x)*.18):0;k.angle+=(angle-k.angle)*Math.min(1,DT*9);
}
// Sweep the ball against the three frame cylinders. The normal is evaluated at
// the first contact, so a glancing shot can keep travelling into the goal.
function goalFrameCollision(previous, ball) {
  const radius = PENALTY_PHYSICS.ballRadius + PENALTY_PHYSICS.postRadius;
  const motion = { x:ball.x-previous.x, y:ball.y-previous.y, z:ball.z-previous.z };
  const contacts = [];
  const sweep = (axis, center, part) => {
    const offset = previous[axis] - center;
    const depth = previous.z - PENALTY_PHYSICS.goalZ;
    const a = motion[axis] ** 2 + motion.z ** 2;
    if (a < 1e-12) return;
    const b = 2 * (offset * motion[axis] + depth * motion.z);
    const c = offset ** 2 + depth ** 2 - radius ** 2;
    const discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return;
    const t = (-b - Math.sqrt(discriminant)) / (2 * a);
    if (t < -1e-7 || t > 1) return;
    const point = { x:lerp(previous.x,ball.x,Math.max(0,t)), y:lerp(previous.y,ball.y,Math.max(0,t)), z:lerp(previous.z,ball.z,Math.max(0,t)) };
    if (axis === 'x' ? point.y < 0 || point.y > PENALTY_PHYSICS.goalHeight : Math.abs(point.x) > PENALTY_PHYSICS.goalWidth / 2) return;
    const normal = { x:0, y:0, z:(point.z-PENALTY_PHYSICS.goalZ)/radius };
    normal[axis] = (point[axis]-center)/radius;
    const approach = motion.x*normal.x + motion.y*normal.y + motion.z*normal.z;
    if (approach >= 0) return;
    contacts.push({ t:Math.max(0,t), point, normal, part });
  };
  sweep('x',-PENALTY_PHYSICS.goalWidth/2,'post');
  sweep('x',PENALTY_PHYSICS.goalWidth/2,'post');
  sweep('y',PENALTY_PHYSICS.goalHeight,'crossbar');
  return contacts.sort((a,b)=>a.t-b.t)[0] || null;
}
/** Deterministic 120 Hz 3D simulation, used directly by the browser game. */
export function simulatePenaltyShot(options={}){
  const mode=PENALTY_MODES[options.mode]||PENALTY_MODES.placement,power=clamp(Number(options.power)||.65,.06,1),inputSpin=clamp(Number(options.spin)||0,-1,1),spin=inputSpin*mode.spinFactor,random=randomGenerator(Number(options.seed)||1827);
  const aim={x:clamp(Number(options.aim?.x)||0,-5.2,5.2),y:clamp(Number(options.aim?.y)||1.25,.03,4)};
  const dispersion=mode.dispersion*(.4+power*power*1.2)*(options.accuracyScale??1);
  const target={x:aim.x+gaussian(random)*dispersion,y:aim.y+gaussian(random)*dispersion*.7};
  const speed=mode.baseSpeed+power*mode.addedSpeed,v=initialVelocity(target,speed),ball={x:0,y:.11,z:0,vx:v.x,vy:v.y,vz:v.z};
  const c={enabled:true,reaction:.18,reactionVariation:.065,error:.22,moveSpeed:3.6,diveSpeed:6.8,acceleration:28,jumpSpeed:2.8,...(options.keeper||{})};
  const k={x:0,y:.86,vx:0,vy:0,angle:0,target:null,diving:false,jumped:false,reaction:c.reaction+random()*c.reactionVariation,errorX:gaussian(random)*c.error,errorY:gaussian(random)*c.error*.65,diveStart:100};
  const frames=[],frameContacts=[];let event=null,crossing=null,bounces=0,travelled=0,time=0;
  const snapshot=()=>({time,ball:{x:ball.x,y:ball.y,z:ball.z},keeper:{x:k.x,y:k.y,angle:k.angle,diving:k.diving},rotation:travelled/.11,event:event?.type||null});frames.push(snapshot());
  for(let step=1;step<=480;step++){
    time=step*DT;const previous={x:ball.x,y:ball.y,z:ball.z},ballSpeed=Math.hypot(ball.vx,ball.vy,ball.vz),drag=PENALTY_PHYSICS.drag*(options.dragScale??1)*ballSpeed;
    ball.vx+=(spin*5.5*Math.abs(ball.vz)/24-drag*ball.vx)*DT;ball.vy+=(-9.81-drag*ball.vy+Math.abs(spin)*.35)*DT;ball.vz-=drag*ball.vz*DT;
    ball.x+=ball.vx*DT;ball.y+=ball.vy*DT;ball.z+=ball.vz*DT;travelled+=Math.hypot(ball.x-previous.x,ball.y-previous.y,ball.z-previous.z);
    if(ball.y<.11){ball.y=.11;ball.vy=Math.abs(ball.vy)>.8?Math.abs(ball.vy)*.37:0;ball.vx*=.93;ball.vz*=.94;bounces++;}
    updateKeeper(k,ball,time,spin,c);
    if(!event&&c.enabled&&ball.vz>0){const contact=keeperCollision(previous,ball,k);if(contact){event={type:'save',time,point:contact.point,part:contact.part};ball.z=contact.point.z-.15;ball.vz=-Math.abs(ball.vz)*.32;ball.vx+=(ball.x-k.x)*2.2;ball.vy=Math.max(.8,Math.abs(ball.vy)*.45);}}
    if(!event){
      const contact=goalFrameCollision(previous,ball);
      if(contact){
        const {normal,point,t,part}=contact;
        const normalSpeed=ball.vx*normal.x+ball.vy*normal.y+ball.vz*normal.z;
        ball.vx=(ball.vx-1.65*normalSpeed*normal.x)*.99;
        ball.vy=(ball.vy-1.65*normalSpeed*normal.y)*.99;
        ball.vz=(ball.vz-1.65*normalSpeed*normal.z)*.99;
        const remaining=DT*(1-t);
        ball.x=point.x+normal.x*.001+ball.vx*remaining;
        ball.y=Math.max(.11,point.y+normal.y*.001+ball.vy*remaining);
        ball.z=point.z+normal.z*.001+ball.vz*remaining;
        frameContacts.push({time:time-DT+DT*t,point:{...point},part});
      }
      const line=PENALTY_PHYSICS.goalZ+PENALTY_PHYSICS.ballRadius;
      if(previous.z<line&&ball.z>=line&&ball.vz>0){
        const t=clamp((line-previous.z)/(ball.z-previous.z),0,1);
        crossing={x:lerp(previous.x,ball.x,t),y:lerp(previous.y,ball.y,t),z:line};
        const inside=Math.abs(crossing.x)<3.55&&crossing.y<2.33;
        event={type:inside?'goal':frameContacts.length?'post':crossing.y>2.44?'high':'wide',time:time-DT+DT*t,point:crossing};
      }else if(frameContacts.length&&ball.vz<0&&ball.z<PENALTY_PHYSICS.goalZ-.45){
        const last=frameContacts.at(-1);event={type:'post',time,point:last.point,part:last.part};
      }
    }
    if(event?.type==='goal'&&ball.z>=12.35){ball.z=12.35;ball.vz=-Math.abs(ball.vz)*.12;ball.vx*=.5;ball.vy*=.4;}
    frames.push(snapshot());if(event&&time>=event.time+.85)break;
    if(!event&&(time>=3||(bounces>4&&ballSpeed<2))){const last=frameContacts.at(-1);event={type:last?'post':'short',time,point:last?.point||{x:ball.x,y:ball.y,z:ball.z},...(last?{part:last.part}:{})};break;}
  }
  event||={type:'short',time,point:{x:ball.x,y:ball.y,z:ball.z}};
  return{frames,event,frameContacts,outcome:event.type,input:{aim,power,mode:options.mode||'placement',spin:inputSpin,seed:Number(options.seed)||1827},stats:{speedKmh:speed*3.6,flightTime:event.time,point:event.point,bounces,reaction:k.reaction,spin}};
}
const arrow='<svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 16 15 5M5 5h10v10" stroke="currentColor" stroke-width="1.3"/></svg>';
const replayIcon='<svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3 7h5V2M3.7 7a7 7 0 1 1-.2 6" stroke="currentColor" stroke-width="1.3"/><path d="m9 7 4 3-4 3Z" fill="currentColor"/></svg>';
const outcomes={goal:'进球',save:'被扑出',post:'击中门框',high:'高出横梁',wide:'偏出球门',short:'未到门线'};
export function renderPenalty(){
  return `<section class="penalty-widget" aria-label="点球游戏">
    <header class="penalty-topline"><div class="penalty-game-label"><span>11 M</span><i></i><span>PENALTY</span></div><div class="penalty-round" role="group" aria-label="本轮五次射门">${Array.from({length:5},(_,i)=>`<span class="penalty-shot" data-penalty-shot="${i}" aria-label="第 ${i+1} 球，未射门"></span>`).join('')}<span class="penalty-round-count" data-penalty-count>0 / 5</span></div><div class="penalty-best">BEST <strong data-penalty-best>${readBest()}</strong><span>/ 5</span></div></header>
    <div class="penalty-stage"><canvas class="penalty-canvas" role="button" tabindex="0" aria-label="点球球场。移动准星瞄准，按住蓄力后松开射门；键盘方向键瞄准，空格蓄力，回车使用标准力度。" aria-describedby="penalty-help">浏览器暂不支持画布。</canvas><div class="penalty-scene-label" aria-hidden="true"><span>STRETFORD END</span><span>7.32 × 2.44 M</span></div><div class="penalty-result" data-penalty-result aria-hidden="true"><strong data-penalty-result-title></strong><span data-penalty-result-detail></span></div><div class="penalty-aim-readout" data-penalty-aim></div><div class="penalty-power"><div><span data-penalty-power-label>力度</span><strong data-penalty-power-value>65<span>%</span></strong></div><div class="penalty-power-track"><span data-penalty-power-fill style="transform:scaleX(.65)"></span><i></i></div><small data-penalty-action-hint>按住蓄力，松开射门</small></div><div class="penalty-scene-caption" aria-hidden="true">MUFC / MATCHDAY</div></div>
    <div class="penalty-controls"><div class="penalty-mode-control"><span class="penalty-control-label">射门方式</span><div class="penalty-modes" role="group" aria-label="射门方式">${Object.entries(PENALTY_MODES).map(([id,m])=>`<button type="button" data-penalty-mode="${id}" aria-pressed="${id==='placement'}" title="${m.detail}">${m.name}</button>`).join('')}</div><span class="penalty-mode-detail" data-penalty-mode-detail>${PENALTY_MODES.placement.detail}</span></div><div class="penalty-spin-control"><label class="penalty-control-label" for="penalty-spin">旋转 <output id="penalty-spin-value" for="penalty-spin">0.0</output></label><div class="penalty-spin-row"><span>左</span><input id="penalty-spin" type="range" min="-1" max="1" step=".1" value="0" aria-label="旋转方向与强度"><span>右</span></div></div><div class="penalty-actions"><button class="penalty-replay" type="button" data-penalty-replay disabled>${replayIcon} 慢速回放</button><button class="penalty-shoot" type="button" data-penalty-shoot>按住蓄力 ${arrow}</button></div></div>
    <div class="penalty-statline"><div><span>射速</span><strong data-penalty-speed>—</strong><em>km/h</em></div><div><span>飞行</span><strong data-penalty-flight>—</strong><em>s</em></div><div class="penalty-landing-stat"><span data-penalty-point-label>门线落点</span><strong data-penalty-landing>—</strong></div><button type="button" class="penalty-reset" data-penalty-reset>重新开始</button></div><p id="penalty-help" class="penalty-help"><span>拖动瞄准 · 按住蓄力 / 松开射门</span><span>方向键瞄准 · 空格蓄力 · Enter 标准力度</span></p><p class="penalty-sr-only" data-penalty-status role="status" aria-live="polite">五球一轮。瞄准球门，按住蓄力后松开射门。</p>
  </section>`;
}
export function initPenalty(){
  const root=document.querySelector('.penalty-widget');if(!root)return()=>{};
  const $=selector=>root.querySelector(selector),canvas=$('.penalty-canvas'),ctx=canvas.getContext('2d'),shootButton=$('[data-penalty-shoot]'),replayButton=$('[data-penalty-replay]'),status=$('[data-penalty-status]'),result=$('[data-penalty-result]'),dots=[...root.querySelectorAll('[data-penalty-shot]')],modeButtons=[...root.querySelectorAll('[data-penalty-mode]')],spinInput=$('#penalty-spin'),reducedMotion=matchMedia('(prefers-reduced-motion: reduce)'),removers=[];
  let activePointerId=null;
  let disposed=false,animation=0,width=1100,height=620,camera,state='ready',mode='placement',spin=0,aim={x:1.8,y:1.5},power=.65,best=readBest(),goals=0,attempts=[],lastShot=null,playbackStart=0,chargeStart=0,chargeSource=null,displayFrame=idleFrame(),pointerShotTime=-1000,aimVisible=true;
  const listen=(target,event,handler,options)=>{target.addEventListener(event,handler,options);removers.push(()=>target.removeEventListener(event,handler,options));};
  function idleFrame(){return{time:0,ball:{x:0,y:.11,z:0},keeper:{x:0,y:.86,angle:0,diving:false},rotation:0,event:null};}
  function project(p){const dy=p.y-camera.y,dz=p.z-camera.z,depth=dz*camera.cos-dy*camera.sin,v=dy*camera.cos+dz*camera.sin;return{x:width*.5+camera.f*p.x/Math.max(.2,depth),y:height*.28-camera.f*v/Math.max(.2,depth),scale:camera.f/Math.max(.2,depth)};}
  function aimFromPointer(event){const b=canvas.getBoundingClientRect(),sx=(event.clientX-b.left-width*.5)/camera.f,sy=(height*.28-(event.clientY-b.top))/camera.f,dy=sy*camera.cos-camera.sin,dz=sy*camera.sin+camera.cos,t=(11-camera.z)/Math.max(.01,dz);aim={x:clamp(sx*t,-5.2,5.2),y:clamp(camera.y+dy*t,.03,4)};aimVisible=true;updateAim();}
  function path(points,stroke,lineWidth=1,close=false,fill=null){ctx.beginPath();points.forEach((p,i)=>{const q=project(p);if(!i)ctx.moveTo(q.x,q.y);else ctx.lineTo(q.x,q.y);});if(close)ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lineWidth;ctx.stroke();}}
  function drawStadium(){
    const sky=ctx.createLinearGradient(0,0,0,height);sky.addColorStop(0,'#0a111b');sky.addColorStop(.5,'#202b31');sky.addColorStop(1,'#12221d');ctx.fillStyle=sky;ctx.fillRect(0,0,width,height);
    const haze=ctx.createRadialGradient(width*.5,height*.17,0,width*.5,height*.17,width*.66);haze.addColorStop(0,'#b1404623');haze.addColorStop(.5,'#a3453910');haze.addColorStop(1,'#762f3200');ctx.fillStyle=haze;ctx.fillRect(0,0,width,height*.7);
    ctx.fillStyle='#080e13';ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(width,0);ctx.lineTo(width,height*.11);ctx.lineTo(width*.5,height*.05);ctx.lineTo(0,height*.11);ctx.fill();
    for(let tier=0;tier<4;tier++){const y=height*(.12+tier*.055);ctx.strokeStyle=tier===3?'#d3c9b82e':'#a3a9a31a';ctx.lineWidth=tier===3?2:1;ctx.beginPath();ctx.moveTo(0,y+height*.04);ctx.quadraticCurveTo(width*.5,y-height*.025,width,y+height*.04);ctx.stroke();for(let row=0;row<3;row++)for(let seat=0;seat<115;seat++){const x=width*seat/114,curve=((x/width-.5)**2)*height*.11,n=Math.abs(Math.sin(tier*17.3+row*8.5+seat*31.1));ctx.fillStyle=n>.8?'#c7b3a461':n>.45?'#a4424243':'#8f999c26';ctx.fillRect(x,y+row*height*.013+curve,Math.max(1,width*.0017),Math.max(1,height*.0024));}}
    const ground=ctx.createLinearGradient(0,height*.32,0,height);ground.addColorStop(0,'#365548');ground.addColorStop(.45,'#263e30');ground.addColorStop(1,'#172a22');ctx.fillStyle=ground;ctx.fillRect(0,project({x:0,y:0,z:32}).y,width,height);
    for(let z=-4;z<=34;z+=2)path([{x:-34,y:0,z},{x:34,y:0,z},{x:34,y:0,z:z+2},{x:-34,y:0,z:z+2}],null,0,true,(z/2)%2?'#c1d4a806':'#00000005');
    const pitchLight=ctx.createRadialGradient(width*.5,height*.62,0,width*.5,height*.62,width*.65);pitchLight.addColorStop(0,'#bbc39c17');pitchLight.addColorStop(1,'#bbc39c00');ctx.fillStyle=pitchLight;ctx.fillRect(0,0,width,height);
    const lineWidth=Math.max(.8,width/950);path([{x:-28,y:.006,z:11},{x:28,y:.006,z:11}],'#ecede04f',lineWidth);path([{x:-9.16,y:.006,z:11},{x:-9.16,y:.006,z:5.5},{x:9.16,y:.006,z:5.5},{x:9.16,y:.006,z:11}],'#e8ead953',lineWidth);path([{x:-20.16,y:.006,z:11},{x:-20.16,y:.006,z:-5.5},{x:20.16,y:.006,z:-5.5},{x:20.16,y:.006,z:11}],'#e8ead93a',lineWidth);
    const spot=project({x:0,y:.003,z:0});ctx.fillStyle='#eaeadd8c';ctx.beginPath();ctx.ellipse(spot.x,spot.y,spot.scale*.09,spot.scale*.035,0,0,Math.PI*2);ctx.fill();
    for(const side of[-1,1]){const x=width*(side<0?.08:.92),y=height*.10,lamp=ctx.createRadialGradient(x,y,0,x,y,width*.28);lamp.addColorStop(0,'#eef4e821');lamp.addColorStop(.15,'#eef4e80a');lamp.addColorStop(1,'#eef4e800');ctx.fillStyle=lamp;ctx.fillRect(0,0,width,height);ctx.fillStyle='#e6ebd8b0';ctx.shadowColor='#f4ffe3';ctx.shadowBlur=width*.016;for(let i=0;i<6;i++)ctx.fillRect(x+i*width*.006-width*.016,y,width*.003,height*.006);ctx.shadowBlur=0;}
    const shade=ctx.createRadialGradient(width*.5,height*.5,width*.15,width*.5,height*.5,width*.8);shade.addColorStop(0,'#030b0800');shade.addColorStop(1,'#030b088f');ctx.fillStyle=shade;ctx.fillRect(0,0,width,height);
  }
  function netPoint(x,y,z,time){if(!lastShot||lastShot.outcome!=='goal'||time<lastShot.event.time)return{x,y,z};const age=time-lastShot.event.time-.045;if(age<=0)return{x,y,z};const d=Math.hypot(x-lastShot.event.point.x,y-lastShot.event.point.y);return{x,y,z:z+.22*Math.sin(Math.min(1,age/.12)*Math.PI*.75)*Math.exp(-age*4-d*1.8)};}
  function drawNet(time){const w=3.66,h=2.44,z=11,back=12.45;path([{x:-w,y:0,z:back},{x:-w,y:h,z:back},{x:w,y:h,z:back},{x:w,y:0,z:back}],'#e9eddf25',.8,false,'#07100b14');for(let x=-w;x<=w+.001;x+=w/15){path([netPoint(x,0,back,time),netPoint(x,h,back,time)],'#dde4d638',.65);path([{x,y:h,z},netPoint(x,h,back,time)],'#dde4d62b',.6);}for(let y=0;y<=h+.001;y+=h/11){const points=[];for(let x=-w;x<=w+.001;x+=w/15)points.push(netPoint(x,y,back,time));path(points,'#e2e9da3a',.65);for(const side of[-w,w])path([{x:side,y,z},netPoint(side,y,back,time)],'#dde4d62b',.6);}for(let d=z;d<=back+.001;d+=1.45/6)for(const side of[-w,w])path([{x:side,y:0,z:d},{x:side,y:h,z:d}],'#dde4d62b',.6);}
  function drawFrame(){const w=3.66,h=2.44,z=11,line=Math.max(2.4,project({x:0,y:1,z}).scale*.06*1.8);ctx.lineJoin='round';ctx.lineCap='round';ctx.shadowColor='#0009';ctx.shadowBlur=10;ctx.shadowOffsetY=5;path([{x:-w,y:0,z},{x:-w,y:h,z},{x:w,y:h,z},{x:w,y:0,z}],'#a7b0a5',line+1.4);ctx.shadowBlur=0;ctx.shadowOffsetY=0;path([{x:-w,y:0,z},{x:-w,y:h,z},{x:w,y:h,z},{x:w,y:0,z}],'#e1e5d8',line*.57);}
  function drawKeeper(frame){
    const s=keeperSkeleton(frame.keeper),z=10.62,shadow=project({x:frame.keeper.x,y:0,z});ctx.fillStyle='#02080558';ctx.beginPath();ctx.ellipse(shadow.x,shadow.y,shadow.scale*.46,shadow.scale*.09,0,0,Math.PI*2);ctx.fill();
    const limb=(a,b,r,color)=>{const p=project({...a,z}),q=project({...b,z});ctx.lineWidth=Math.max(1.5,p.scale*r*2);ctx.lineCap='round';ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();};
    limb(s.hip,s.leftKnee,.1,'#26322c');limb(s.hip,s.rightKnee,.1,'#26322c');limb(s.leftKnee,s.leftFoot,.075,'#b9bca8');limb(s.rightKnee,s.rightFoot,.075,'#b9bca8');limb(s.hip,s.shoulder,.195,'#c4bb8f');limb(s.shoulder,s.leftElbow,.075,'#bdb88f');limb(s.leftElbow,s.leftHand,.07,'#bdb88f');limb(s.shoulder,s.rightElbow,.075,'#bdb88f');limb(s.rightElbow,s.rightHand,.07,'#bdb88f');
    for(const hand of[s.leftHand,s.rightHand]){const p=project({...hand,z});ctx.fillStyle='#e5e7d7';ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2.2,p.scale*.105),0,Math.PI*2);ctx.fill();}
    const head=project({...s.head,z});ctx.fillStyle='#c0ac8d';ctx.beginPath();ctx.ellipse(head.x,head.y,head.scale*.115,head.scale*.14,-frame.keeper.angle,0,Math.PI*2);ctx.fill();const a=project({...s.shoulder,z}),b=project({...s.hip,z});ctx.strokeStyle='#222b226e';ctx.lineWidth=Math.max(1,a.scale*.025);ctx.beginPath();ctx.moveTo(a.x,a.y+a.scale*.07);ctx.lineTo(b.x,b.y-a.scale*.07);ctx.stroke();
  }
  function drawBall(frame){
    const ball=frame.ball,p=project(ball),shadow=project({x:ball.x,y:.005,z:ball.z}),r=Math.max(2,p.scale*.11);ctx.fillStyle=`rgba(0,8,3,${clamp(.45-ball.y*.07,.12,.4)})`;ctx.beginPath();ctx.ellipse(shadow.x+r*.3,shadow.y,r*(1.3+ball.y*.2),r*.38,0,0,Math.PI*2);ctx.fill();
    if(lastShot&&(state==='flight'||state==='replay')){const index=Math.round(frame.time/DT);for(let i=5;i>=1;i--){const previous=lastShot.frames[Math.max(0,index-i*2)];if(!previous||previous.time===frame.time)continue;const pp=project(previous.ball);ctx.fillStyle=`rgba(228,231,210,${.075*(1-i/7)})`;ctx.beginPath();ctx.arc(pp.x,pp.y,Math.max(1,pp.scale*.09),0,Math.PI*2);ctx.fill();}}
    const sphere=ctx.createRadialGradient(p.x-r*.35,p.y-r*.4,r*.1,p.x,p.y,r);sphere.addColorStop(0,'#fcfcf3');sphere.addColorStop(.65,'#d8ddcf');sphere.addColorStop(1,'#747f73');ctx.fillStyle=sphere;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();
    const rotation=frame.rotation*.35;ctx.save();ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.clip();for(let panel=0;panel<6;panel++){const longitude=panel*Math.PI/3+rotation,px=p.x+Math.sin(longitude)*r*.8,py=p.y+Math.cos(longitude+rotation*.25)*r*.35;ctx.fillStyle=panel%2?'#284135':'#495b45';ctx.beginPath();for(let corner=0;corner<5;corner++){const a=corner*Math.PI*.4+rotation,x=px+Math.cos(a)*r*.29,y=py+Math.sin(a)*r*.29;if(!corner)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.closePath();ctx.fill();}ctx.restore();
  }
  function drawAim(){if(!aimVisible||!['ready','charging','result'].includes(state))return;const p=project({...aim,z:11}),inside=Math.abs(aim.x)<3.49&&aim.y<2.27,r=clamp(width*.009,7,11);ctx.strokeStyle=inside?'#f0e8c9c9':'#e38583c9';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(p.x-r-5,p.y);ctx.lineTo(p.x-r+2,p.y);ctx.moveTo(p.x+r-2,p.y);ctx.lineTo(p.x+r+5,p.y);ctx.moveTo(p.x,p.y-r-5);ctx.lineTo(p.x,p.y-r+2);ctx.moveTo(p.x,p.y+r-2);ctx.lineTo(p.x,p.y+r+5);ctx.stroke();const b=project({x:0,y:.11,z:0});ctx.strokeStyle='#e7e2ca14';ctx.lineWidth=.8;ctx.setLineDash([3,7]);ctx.beginPath();ctx.moveTo(b.x,b.y-14);ctx.lineTo(p.x,p.y);ctx.stroke();ctx.setLineDash([]);}
  function draw(){if(!ctx||disposed||!camera||width<1||height<1)return;const ratio=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);}ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);drawStadium();drawNet(displayFrame.time);drawKeeper(displayFrame);drawBall(displayFrame);drawFrame();drawAim();}
  function resize(){width=canvas.clientWidth;height=canvas.clientHeight;camera={y:3.3,z:-8,f:Math.min(width*1.6,height*3),sin:Math.sin(.13),cos:Math.cos(.13)};draw();}
  function updateAim(){$('[data-penalty-aim]').textContent=`瞄准：${Math.abs(aim.x)<.05?'中路':`${aim.x<0?'左':'右'} ${Math.abs(aim.x).toFixed(1)} m`} / 高 ${aim.y.toFixed(1)} m`;}
  function updatePower(){$('[data-penalty-power-value]').innerHTML=`${Math.round(power*100)}<span>%</span>`;$('[data-penalty-power-fill]').style.transform=`scaleX(${power})`;$('[data-penalty-power-fill]').dataset.high=String(power>.88);$('[data-penalty-power-label]').textContent=state==='charging'?'蓄力中':'力度';$('[data-penalty-action-hint]').textContent=state==='charging'?'松开射门':state==='replay'?'0.35 × 慢速回放':'按住蓄力，松开射门';}
  function updateControls(){
    const locked=['flight','replay','charging'].includes(state),over=attempts.length>=5;root.dataset.state=state;$('[data-penalty-count]').textContent=`${attempts.length} / 5`;
    dots.forEach((dot,i)=>{dot.dataset.result=attempts[i]||'';dot.setAttribute('aria-label',`第 ${i+1} 球，${outcomes[attempts[i]]||'未射门'}`);});
    modeButtons.forEach(button=>{button.disabled=locked;button.setAttribute('aria-pressed',String(button.dataset.penaltyMode===mode));});spinInput.disabled=locked;replayButton.disabled=!lastShot||locked;$('[data-penalty-reset]').disabled=locked;shootButton.disabled=state==='flight'||state==='replay';
    shootButton.innerHTML=over?`再来一轮 ${arrow}`:state==='charging'?`松开射门 ${arrow}`:state==='flight'?'射门中':state==='replay'?'回放中':`按住蓄力 ${arrow}`;canvas.setAttribute('aria-disabled',String(state==='flight'||state==='replay'||over));updatePower();
  }
  function showStats(shot){$('[data-penalty-speed]').textContent=Math.round(shot.stats.speedKmh);$('[data-penalty-flight]').textContent=shot.stats.flightTime.toFixed(2);$('[data-penalty-point-label]').textContent=shot.outcome==='save'?'扑救接触点':shot.outcome==='post'?'门框接触点':'门线落点';const p=shot.stats.point;$('[data-penalty-landing]').textContent=`${Math.abs(p.x)<.05?'中路':`${p.x<0?'左':'右'} ${Math.abs(p.x).toFixed(2)} m`} / 高 ${p.y.toFixed(2)} m`;}
  function showResult(shot,replay=false){$('[data-penalty-result-title]').textContent=replay?'慢速回放':attempts.length===5?`${goals} / 5`:outcomes[shot.outcome];$('[data-penalty-result-detail]').textContent=replay?'0.35 ×':attempts.length===5?`本轮进球 · ${outcomes[shot.outcome]}`:`${Math.round(shot.stats.speedKmh)} km/h · ${shot.stats.flightTime.toFixed(2)} s`;result.classList.add('is-visible');result.dataset.outcome=shot.outcome;}
  function finishPlayback(isReplay){
    if(disposed||!lastShot)return;displayFrame=lastShot.frames.at(-1);
    if(!isReplay){attempts.push(lastShot.outcome);if(lastShot.outcome==='goal')goals++;if(goals>best){best=goals;$('[data-penalty-best]').textContent=best;try{localStorage.setItem(BEST_KEY,String(best));}catch{}}showStats(lastShot);status.textContent=`${outcomes[lastShot.outcome]}。球速 ${Math.round(lastShot.stats.speedKmh)} 公里每小时，飞行 ${lastShot.stats.flightTime.toFixed(2)} 秒。${attempts.length===5?`本轮五球进 ${goals} 球。`:`已完成 ${attempts.length} 次射门。`}`;}
    state='result';aimVisible=false;showResult(lastShot);updateControls();draw();
  }
  function animate(now){
    if(disposed)return;
    if(state==='charging'){power=clamp(.06+(now-chargeStart)/1300,.06,1);updatePower();draw();animation=requestAnimationFrame(animate);}
    else if(state==='flight'||state==='replay'){const isReplay=state==='replay',time=(now-playbackStart)/1000*(isReplay?.35:1),i=clamp(Math.floor(time/DT),0,lastShot.frames.length-1);displayFrame=lastShot.frames[i];draw();if(time>=lastShot.frames.at(-1).time){animation=0;finishPlayback(isReplay);}else animation=requestAnimationFrame(animate);}
  }
  function releaseShot(releasePower=power){
    if(disposed||attempts.length>=5||['flight','replay'].includes(state))return;cancelAnimationFrame(animation);animation=0;power=clamp(releasePower,.06,1);chargeSource=null;activePointerId=null;
    lastShot=simulatePenaltyShot({aim,power,mode,spin,seed:freshSeed()});displayFrame=lastShot.frames[0];state='flight';aimVisible=false;result.classList.remove('is-visible');status.textContent=`${PENALTY_MODES[mode].name}，力度 ${Math.round(power*100)}%。射门。`;updateControls();
    if(reducedMotion.matches||document.hidden){finishPlayback(false);return;}playbackStart=performance.now();animation=requestAnimationFrame(animate);
  }
  function startCharge(source){if(disposed||['flight','replay','charging'].includes(state))return;if(attempts.length>=5){reset();return;}state='charging';chargeSource=source;chargeStart=performance.now();power=.06;displayFrame=idleFrame();aimVisible=true;result.classList.remove('is-visible');updateControls();draw();animation=requestAnimationFrame(animate);}
  function cancelCharge(){if(state!=='charging')return;cancelAnimationFrame(animation);animation=0;state='ready';chargeSource=null;activePointerId=null;power=.65;updateControls();draw();}
  function reset(){cancelAnimationFrame(animation);animation=0;attempts=[];goals=0;lastShot=null;state='ready';chargeSource=null;activePointerId=null;power=.65;aim={x:1.8,y:1.5};displayFrame=idleFrame();aimVisible=true;result.classList.remove('is-visible');$('[data-penalty-speed]').textContent='—';$('[data-penalty-flight]').textContent='—';$('[data-penalty-landing]').textContent='—';status.textContent='新一轮。五次射门。';updateAim();updateControls();draw();}
  function replay(){if(!lastShot||['flight','replay','charging'].includes(state))return;if(reducedMotion.matches){showResult(lastShot);status.textContent='已显示上一球结果。';return;}state='replay';displayFrame=lastShot.frames[0];aimVisible=false;showResult(lastShot,true);updateControls();playbackStart=performance.now();animation=requestAnimationFrame(animate);}
  if(!ctx){shootButton.disabled=true;status.textContent='浏览器暂不支持画布。';return()=>{};}
  for(const button of modeButtons)listen(button,'click',()=>{if(['flight','replay','charging'].includes(state))return;mode=button.dataset.penaltyMode;if(mode==='curl'&&Math.abs(spin)<.1){spin=.6;spinInput.value='.6';$('#penalty-spin-value').textContent='+0.6';}$('[data-penalty-mode-detail]').textContent=PENALTY_MODES[mode].detail;updateControls();});
  listen(spinInput,'input',()=>{spin=Number(spinInput.value);$('#penalty-spin-value').textContent=`${spin>0?'+':''}${spin.toFixed(1)}`;});
  listen(canvas,'pointermove',event=>{if(event.isPrimary===false||(activePointerId!==null&&event.pointerId!==activePointerId))return;if(!['ready','charging','result'].includes(state)||attempts.length>=5||(event.pointerType==='touch'&&state!=='charging'))return;aimFromPointer(event);if(state!=='charging'){displayFrame=idleFrame();result.classList.remove('is-visible');draw();}});
  listen(canvas,'pointerdown',event=>{if(event.isPrimary===false||event.button!==0||['flight','replay'].includes(state)||attempts.length>=5)return;event.preventDefault();canvas.focus({preventScroll:true});canvas.setPointerCapture?.(event.pointerId);aimFromPointer(event);startCharge('canvas');if(state==='charging'&&chargeSource==='canvas')activePointerId=event.pointerId;});
  listen(canvas,'pointerup',event=>{if(event.isPrimary===false||event.pointerId!==activePointerId||state!=='charging'||chargeSource!=='canvas')return;aimFromPointer(event);power=clamp(.06+(performance.now()-chargeStart)/1300,.06,1);pointerShotTime=performance.now();releaseShot();});
  listen(canvas,'pointercancel',event=>{if(event.pointerId===activePointerId)cancelCharge();});
  listen(shootButton,'pointerdown',event=>{if(event.isPrimary===false||event.button!==0||shootButton.disabled)return;event.preventDefault();shootButton.setPointerCapture?.(event.pointerId);startCharge('button');if(state==='charging'&&chargeSource==='button')activePointerId=event.pointerId;});
  listen(shootButton,'pointerup',event=>{if(event.isPrimary===false||event.pointerId!==activePointerId||state!=='charging'||chargeSource!=='button')return;power=clamp(.06+(performance.now()-chargeStart)/1300,.06,1);pointerShotTime=performance.now();releaseShot();});
  listen(shootButton,'pointercancel',event=>{if(event.pointerId===activePointerId)cancelCharge();});
  listen(shootButton,'click',event=>{if(performance.now()-pointerShotTime<400)return;if(attempts.length>=5){reset();return;}if(event.detail===0&&state!=='charging')releaseShot(.65);});
  const onKeyDown=event=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter',' ','Escape'].includes(event.key))return;event.preventDefault();if(event.key==='Escape'){cancelCharge();return;}if(attempts.length>=5){if((event.key==='Enter'||event.key===' ')&&!event.repeat)reset();return;}if(['flight','replay'].includes(state))return;
    if(event.key===' '){if(!event.repeat)startCharge('keyboard');return;}if(event.key==='Enter'){if(!event.repeat)releaseShot(.65);return;}
    const step=event.shiftKey?.07:.25;if(event.key==='ArrowLeft')aim.x=clamp(aim.x-step,-5.2,5.2);if(event.key==='ArrowRight')aim.x=clamp(aim.x+step,-5.2,5.2);if(event.key==='ArrowUp')aim.y=clamp(aim.y+step,.03,4);if(event.key==='ArrowDown')aim.y=clamp(aim.y-step,.03,4);aimVisible=true;updateAim();if(state!=='charging'){displayFrame=idleFrame();result.classList.remove('is-visible');draw();}status.textContent=$('[data-penalty-aim]').textContent;
  };
  listen(canvas,'keydown',onKeyDown);listen(shootButton,'keydown',onKeyDown);
  for(const target of[canvas,shootButton])listen(target,'keyup',event=>{if(event.key===' '&&state==='charging'&&chargeSource==='keyboard'){event.preventDefault();power=clamp(.06+(performance.now()-chargeStart)/1300,.06,1);releaseShot();}});
  listen(canvas,'click',event=>{if(event.detail===0&&performance.now()-pointerShotTime>400&&state!=='charging')releaseShot(.65);});
  listen(canvas,'blur',()=>{if(chargeSource==='keyboard')cancelCharge();});listen(shootButton,'blur',()=>{if(chargeSource==='keyboard')cancelCharge();});listen(replayButton,'click',replay);listen($('[data-penalty-reset]'),'click',reset);
  listen(document,'visibilitychange',()=>{if(!document.hidden)return;if(state==='charging')cancelCharge();else if(state==='flight'||state==='replay'){const wasReplay=state==='replay';cancelAnimationFrame(animation);animation=0;finishPlayback(wasReplay);}});
  listen(reducedMotion,'change',()=>{if(reducedMotion.matches&&(state==='flight'||state==='replay')){const wasReplay=state==='replay';cancelAnimationFrame(animation);animation=0;finishPlayback(wasReplay);}});
  listen(window,'storage',event=>{if(event.key===BEST_KEY||event.key===null){best=readBest();$('[data-penalty-best]').textContent=best;}});
  let observer;if(typeof ResizeObserver==='function'){observer=new ResizeObserver(resize);observer.observe(canvas);}else listen(window,'resize',resize);
  updateAim();updateControls();resize();return()=>{disposed=true;cancelAnimationFrame(animation);observer?.disconnect();removers.forEach(remove=>remove());};
}
