const $=id=>document.getElementById(id);

const presetSelect=$("presetSelect");
const recognizeBtn=$("recognizeBtn");
const audioPlayer=$("audioPlayer");
const specCanvas=$("specCanvas");
const freqCanvas=$("freqCanvas");
const specIdle=$("specIdle");
const specScan=$("specScan");

let presets=[];
let profileCache=null;
let audioBuffer=null;
let audioStats=null;
let specData=null;
let runId=0;
let ttsText="";
let eventQueue=[];
let eventPlaying=false;
let streamPaused=false;
let lastResultText="";
let eventHistory=[];
let currentView="dual";

const TOTAL_MS=14800;

const STAGES=[
  {at:0,title:"任务接入"},
  {at:850,title:"音频质量检查"},
  {at:2050,title:"语音时频分析"},
  {at:5150,title:"个性化语音表征提取"},
  {at:7850,title:"个性化语音库匹配"},
  {at:10850,title:"时序一致性校验"},
  {at:12650,title:"拒识策略判断"},
  {at:13650,title:"规范中文结果生成"}
];

const SUB_EVENTS=[
  {at:120,text:"已接收王灏真实语音任务",cls:"ok",state:"已接收"},
  {at:640,text:"正在读取音频文件与基础参数",state:"读取中"},
  {at:1180,text:"完成输入格式检查，准备语音质量分析",state:"已检查"},
  {at:1710,text:"正在计算语音能量与有效动态范围",state:"分析中"},
  {at:2460,text:"正在生成真实录音的简化 STFT 时频结构",state:"分析中"},
  {at:3420,text:"时频结构持续生成，分析音频能量变化",state:"分析中"},
  {at:4590,text:"时频分析阶段完成，进入语音表征提取",cls:"ok",state:"已完成"},
  {at:5450,text:"正在提取个性化语音表征",state:"提取中"},
  {at:6420,text:"正在执行表征归一化与时序特征准备",state:"提取中"},
  {at:7960,text:"正在加载王灏个性化语音记忆",state:"加载中"},
  {at:8620,text:"已注册规范表达与个性化样本加载完成",cls:"ok",state:"已就绪"},
  {at:9250,text:"正在执行候选规范表达匹配",state:"匹配中"},
  {at:10680,text:"候选匹配阶段完成，进入时序一致性复核",state:"匹配中"},
  {at:11240,text:"正在执行时序特征校验",state:"校验中"},
  {at:12150,text:"正在执行已注册 / 未注册表达拒识判断",state:"判断中"},
  {at:13380,text:"拒识判断完成，准备生成规范中文",cls:"ok",state:"已完成"},
  {at:13750,text:"正在提交最终个性化识别请求",cls:"warn",state:"等待"}
];

async function api(url,opt={}){
  const r=await fetch(url,opt);
  if(!r.ok){
    let m=`${r.status} ${r.statusText}`;
    try{m=(await r.json()).detail||m}catch{}
    throw new Error(m);
  }
  return r.json();
}

const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const esc=s=>String(s)
  .replaceAll("&","&amp;")
  .replaceAll("<","&lt;")
  .replaceAll(">","&gt;")
  .replaceAll('"',"&quot;");

function tickClock(){
  const d=new Date();
  $("clockDate").textContent=d.toLocaleDateString("zh-CN",{
    year:"numeric",month:"2-digit",day:"2-digit",weekday:"short"
  });
  $("clockTime").textContent=d.toLocaleTimeString("zh-CN",{hour12:false});
}
setInterval(tickClock,1000);
tickClock();

async function boot(){
  try{
    const [h,p,q]=await Promise.all([
      api("/api/health"),
      api("/api/profile"),
      api("/api/presets")
    ]);

    profileCache=p;
    presets=q.presets||[];

    $("systemStatus").textContent="运行正常";
    $("modelState").textContent=h.model_loaded?"模型已加载":"模型加载中";
    $("studentName").textContent=p.student_name;
    $("registeredCount").textContent=p.registered_expression_count;
    $("sampleCount").textContent=p.registered_expression_count*p.enrollment_per_expression;
    $("taskRegistered").textContent=p.registered_expression_count;
    $("taskSamples").textContent=p.registered_expression_count*p.enrollment_per_expression;

    fillPresets();
    generateTaskId();
    bindInteractiveControls();

    if(!supportsTts()){
      $("autoTts").checked=false;
      $("autoTts").disabled=true;
    }
  }catch(e){
    $("systemStatus").textContent="初始化异常";
    console.error(e);
  }
}

function bindInteractiveControls(){
  $("playOriginalBtn").addEventListener("click",()=>{
    if(audioPlayer.paused){
      audioPlayer.play();
      $("playOriginalBtn").textContent="暂停原始语音";
    }else{
      audioPlayer.pause();
      $("playOriginalBtn").textContent="播放原始语音";
    }
  });

  audioPlayer.addEventListener("ended",()=>{
    $("playOriginalBtn").textContent="播放原始语音";
  });

  $("restartBtn").addEventListener("click",()=>{
    if(!recognizeBtn.disabled){
      recognizeBtn.click();
    }
  });

  $("clearStreamBtn").addEventListener("click",()=>{
    eventHistory=[];
    $("eventStream").innerHTML=
      '<div class="event muted static-event"><time>--:--:--</time><span>处理流水已清空</span><b>已清空</b></div>';
  });

  $("exportStreamBtn").addEventListener("click",exportStream);

  $("copyResultBtn").addEventListener("click",async()=>{
    if(!lastResultText)return;
    try{
      await navigator.clipboard.writeText(lastResultText);
      $("copyResultBtn").textContent="已复制";
      setTimeout(()=>$("copyResultBtn").textContent="复制结果",900);
    }catch{
      $("copyResultBtn").textContent="复制失败";
      setTimeout(()=>$("copyResultBtn").textContent="复制结果",900);
    }
  });

  $("fullscreenBtn").addEventListener("click",async()=>{
    if(!document.fullscreenElement){
      await document.documentElement.requestFullscreen?.();
      $("fullscreenBtn").textContent="退出全屏";
    }else{
      await document.exitFullscreen?.();
      $("fullscreenBtn").textContent="全屏展示";
    }
  });

  $("pauseStreamBtn").addEventListener("click",()=>{
    streamPaused=!streamPaused;
    $("pauseStreamBtn").textContent=streamPaused?"继续自动滚动":"暂停自动滚动";
    $("streamState").textContent=streamPaused?"已暂停":"实时";
    if(!streamPaused)scrollStreamToBottom();
  });

  document.querySelectorAll(".view-btn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(".view-btn").forEach(x=>x.classList.remove("active"));
      btn.classList.add("active");
      currentView=btn.dataset.view;
      const dual=$("dualAnalysis");
      dual.className=`dual-analysis view-${currentView}`;
      drawSpectrogram(currentSpectrumProgress());
      drawFrequencySpectrum(currentSpectrumProgress());
    });
  });
}

function fillPresets(){
  presetSelect.innerHTML='<option value="">请选择一条王灏真实录音</option>';

  [
    ["recognition","个性化识别演示"],
    ["rejection","未注册表达拒识演示"]
  ].forEach(([kind,label])=>{
    const g=document.createElement("optgroup");
    g.label=label;

    presets
      .filter(x=>x.demo_kind===kind)
      .forEach(x=>{
        const o=document.createElement("option");
        o.value=x.id;
        o.textContent=x.display_name;
        g.appendChild(o);
      });

    if(g.children.length)presetSelect.appendChild(g);
  });
}

function generateTaskId(){
  const d=new Date();
  const y=d.getFullYear();
  const m=String(d.getMonth()+1).padStart(2,"0");
  const day=String(d.getDate()).padStart(2,"0");
  const suffix=String(Math.floor(100+Math.random()*900));
  $("taskId").textContent=`WH-${y}${m}${day}-${suffix}`;
}

presetSelect.addEventListener("change",async()=>{
  stopTts();
  resetAll();
  generateTaskId();

  const item=presets.find(x=>x.id===presetSelect.value);

  if(!item){
    audioPlayer.removeAttribute("src");
    audioPlayer.load();
    recognizeBtn.disabled=true;
    $("playOriginalBtn").disabled=true;
    $("restartBtn").disabled=true;
    $("taskStatus").textContent="等待接入";
    $("taskBadge").textContent="待接入";
    clearAudioStats();
    return;
  }

  audioPlayer.src=item.audio_url;
  audioPlayer.load();

  recognizeBtn.disabled=false;
  $("playOriginalBtn").disabled=false;
  $("restartBtn").disabled=false;

  $("taskStatus").textContent="已就绪";
  $("taskBadge").textContent="已就绪";

  try{
    audioBuffer=await decodeAudio(item.audio_url);
    audioStats=calculateAudioStats(audioBuffer);
    renderAudioStats(audioStats);

    specData=buildSpectrogram(audioBuffer);
    $("nyquistLabel").textContent=
      `${Math.round(audioBuffer.sampleRate/2000)} kHz`;

    drawFrequencySpectrum(0);
  }catch(e){
    audioBuffer=null;
    audioStats=null;
    specData=null;
    clearAudioStats();
    console.warn(e);
  }
});

recognizeBtn.addEventListener("click",async()=>{
  if(!presetSelect.value)return;

  stopTts();
  resetAll(false);

  const token=++runId;

  recognizeBtn.disabled=true;
  presetSelect.disabled=true;
  $("restartBtn").disabled=true;

  $("taskStatus").textContent="智能分析中";
  $("taskBadge").textContent="处理中";
  $("analysisBadge").textContent="分析中";
  $("decisionBadge").textContent="研判中";
  $("streamState").textContent="实时";

  specIdle.classList.add("hidden");
  specScan.classList.remove("hidden");

  setClosure(0,"done","完成");
  setClosure(1,"active","处理中");

  clearEventQueue();
  eventHistory=[];
  startLiveCursor();

  try{
    if(audioBuffer && !specData){
      specData=buildSpectrogram(audioBuffer);
    }

    const started=performance.now();

    let currentStage=-1;
    let nextSubEvent=0;
    let inferencePromise=null;

    while(true){
      if(token!==runId)return;

      const elapsed=performance.now()-started;
      const pct=Math.min(1,elapsed/TOTAL_MS);

      for(let i=currentStage+1;i<STAGES.length;i++){
        if(elapsed>=STAGES[i].at){
          if(currentStage>=0)markStage(currentStage,"done");
          currentStage=i;
          markStage(i,"active");
          setStageTitle(STAGES[i].title);
        }
      }

      while(
        nextSubEvent<SUB_EVENTS.length &&
        elapsed>=SUB_EVENTS[nextSubEvent].at
      ){
        const ev=SUB_EVENTS[nextSubEvent++];
        queueEvent(
          resolveEventText(ev.text),
          ev.cls||"",
          {
            pause:360+Math.floor(Math.random()*220),
            state:ev.state||"RUN"
          }
        );
      }

      const revealPct=spectrogramReveal(pct);

      drawSpectrogram(revealPct);
      drawFrequencySpectrum(revealPct);

      $("progressText").textContent=
        `${Math.round(pct*100)}%`;

      specScan.style.left=
        `calc(${Math.min(99,revealPct*100)}% - 2px)`;

      updateLiveReadout(revealPct);

      if(!inferencePromise && elapsed>=13750){
        inferencePromise=api(
          `/api/recognize/${encodeURIComponent(presetSelect.value)}`,
          {method:"POST"}
        );
      }

      if(elapsed>=TOTAL_MS)break;

      await sleep(40);
    }

    if(currentStage>=0)markStage(currentStage,"done");

    if(!inferencePromise){
      inferencePromise=api(
        `/api/recognize/${encodeURIComponent(presetSelect.value)}`,
        {method:"POST"}
      );
    }

    setStageTitle("等待最终研判结果");

    queueEvent(
      "展示流程已完成，正在等待真实 Runtime 返回最终结果",
      "warn",
      {pause:900,state:"等待"}
    );

    const result=await inferencePromise;

    if(token!==runId)return;

    drawSpectrogram(1);
    drawFrequencySpectrum(1);

    $("progressText").textContent="100%";
    specScan.classList.add("hidden");

    $("analysisBadge").textContent="已完成";
    $("decisionBadge").textContent="已完成";
    $("taskStatus").textContent="已完成";
    $("taskBadge").textContent="已完成";
    setStageTitle("分析完成");

    queueEvent(
      "真实 Runtime 已返回最终识别结果",
      "ok",
      {pause:700,state:"已完成",final:true}
    );

    await waitEventQueueIdle(token);
    stopLiveCursor();

    await revealResult(result,token);

    setClosure(1,"done","完成");

    if(result.status==="ACCEPT"){
      setClosure(2,"done","完成");
    }else if(result.status==="CONFIRM"){
      setClosure(2,"active","待确认");
    }

    if(result.status==="ACCEPT"&&$("autoTts").checked){
      setTimeout(()=>{
        if(token===runId)speak();
      },320);
    }
  }catch(e){
    if(token===runId){
      specScan.classList.add("hidden");
      $("analysisBadge").textContent="异常";
      $("decisionBadge").textContent="异常";
      $("taskStatus").textContent="异常";

      queueEvent(
        `处理异常：${e.message}`,
        "warn",
        {pause:550,state:"异常"}
      );

      stopLiveCursor();
      showError(e.message);
    }
  }finally{
    if(token===runId){
      recognizeBtn.disabled=false;
      presetSelect.disabled=false;
      $("restartBtn").disabled=false;
    }
  }
});

function resetAll(invalidate=true){
  if(invalidate)runId++;

  [...$("stageTrack").children].forEach(x=>x.className="");

  $("currentStage").textContent="等待任务启动";
  $("progressText").textContent="0%";
  $("analysisBadge").textContent="待开始";
  $("decisionBadge").textContent="待研判";
  $("streamState").textContent=streamPaused?"已暂停":"实时";

  $("eventStream").innerHTML=
    '<div class="event muted static-event"><time>--:--:--</time><span>系统等待任务接入</span><b>WAIT</b></div>';

  $("decisionWaiting").classList.remove("hidden");
  $("decisionResult").classList.add("hidden");

  $("canonicalText").textContent="";
  $("textCursor").classList.add("hidden");

  lastResultText="";
  $("copyResultBtn").disabled=true;

  ttsText="";
  $("speakBtn").disabled=true;
  $("stopSpeakBtn").disabled=true;
  $("ttsBars").classList.add("hidden");

  $("checkText").className="";
  $("checkText").querySelector("b").textContent="待生成";
  $("checkTts").className="";
  $("checkTts").querySelector("b").textContent="待生成";

  setClosure(0,"","就绪");
  setClosure(1,"","等待");
  setClosure(2,"","等待");
  setClosure(3,"","等待");
  setClosure(4,"","下一阶段");

  clearEventQueue();
  stopLiveCursor();

  resetSpectrogram();
  resetFrequencySpectrum();

  $("frameReadout").textContent="0 / 0";
  $("energyReadout").textContent="-- dB";
  $("dominantFreq").textContent="-- Hz";
  $("freqReadout").textContent="0 Hz";
}

function setStageTitle(text){
  const el=$("currentStage");
  el.classList.remove("bump");
  void el.offsetWidth;
  el.textContent=text;
  el.classList.add("bump");
}

function markStage(index,state){
  const el=$("stageTrack").querySelector(`[data-i="${index}"]`);
  if(el)el.className=state;
}

function setClosure(index,state,label){
  const el=$("closureFlow").querySelector(`[data-step="${index}"]`);
  if(!el)return;

  el.className=state;
  el.querySelector("b").textContent=label;
}

function resolveEventText(text){
  if(!audioStats||!profileCache)return text;

  if(text==="完成输入格式检查，准备语音质量分析"){
    return `输入检查完成：${audioStats.sampleRate} Hz / ${audioStats.duration.toFixed(2)} s`;
  }

  if(text==="正在计算语音能量与有效动态范围"){
    return `语音能量分析：RMS ${audioStats.rmsDb.toFixed(1)} dBFS / Peak ${audioStats.peakDb.toFixed(1)} dBFS`;
  }

  if(text==="已注册规范表达与个性化样本加载完成"){
    return `个性化语音库加载完成：${profileCache.registered_expression_count} 项规范表达 / ${profileCache.registered_expression_count*profileCache.enrollment_per_expression} 条样本`;
  }

  return text;
}

/* -----------------------------
   Process trace
----------------------------- */

function clearEventQueue(){
  eventQueue=[];
  eventPlaying=false;
}

function queueEvent(text,cls="",options={}){
  eventQueue.push({
    text,
    cls,
    pause:options.pause??500,
    state:options.state??"处理中",
    final:Boolean(options.final)
  });

  if(!eventPlaying){
    playEventQueue();
  }
}

async function playEventQueue(){
  eventPlaying=true;

  while(eventQueue.length){
    const item=eventQueue.shift();
    await appendEvent(item);
    await sleep(item.pause);
  }

  eventPlaying=false;
}

async function appendEvent(item){
  const now=new Date().toLocaleTimeString("zh-CN",{hour12:false});

  const row=document.createElement("div");
  row.className=`event ${item.cls} entering`;
  row.innerHTML=
    `<time>${now}</time><span></span><b>${esc(item.state)}</b>`;

  $("eventStream").appendChild(row);

  requestAnimationFrame(()=>{
    requestAnimationFrame(()=>{
      row.classList.add("show");
    });
  });

  const span=row.querySelector("span");
  span.classList.add("typing");

  const charDelay=item.final?30:15;

  for(const ch of item.text){
    span.textContent+=ch;
    await sleep(charDelay);
  }

  span.classList.remove("typing");

  eventHistory.push({
    time:now,
    text:item.text,
    state:item.state
  });

  if(!streamPaused){
    scrollStreamToBottom();
  }
}

function startLiveCursor(){
  stopLiveCursor();

  const row=document.createElement("div");
  row.id="liveCursor";
  row.className="event-live";
  row.innerHTML=
    '<time>LIVE</time><span>系统持续处理 <i></i><i></i><i></i></span><b>处理中</b>';

  $("eventStream").appendChild(row);
}

function stopLiveCursor(){
  const el=$("liveCursor");
  if(el)el.remove();
}

function scrollStreamToBottom(){
  const stream=$("eventStream");
  stream.scrollTop=stream.scrollHeight;
}

async function waitEventQueueIdle(token){
  while(eventPlaying||eventQueue.length){
    if(token!==runId)return;
    await sleep(70);
  }
}

function exportStream(){
  const lines=[
    "PAPR-SSL Demo Process Trace",
    `Task: ${$("taskId").textContent}`,
    `Student: ${$("studentName").textContent}`,
    "",
    ...eventHistory.map(
      x=>`${x.time}\t${x.state}\t${x.text}`
    )
  ];

  const blob=new Blob(
    [lines.join("\n")],
    {type:"text/plain;charset=utf-8"}
  );

  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url;
  a.download=`${$("taskId").textContent}_process_trace.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

/* -----------------------------
   Result
----------------------------- */

async function revealResult(result,token){
  $("decisionWaiting").classList.add("hidden");
  $("decisionResult").classList.remove("hidden");

  $("resultBadge").textContent=result.display_status;
  $("latencyText").textContent=
    `模型实际耗时 ${Number(result.latency_ms).toFixed(1)} ms`;

  const text=result.display_text||"";

  $("canonicalText").textContent="";
  $("textCursor").classList.remove("hidden");

  for(const ch of text){
    if(token!==runId)return;
    $("canonicalText").textContent+=ch;
    await sleep(88);
  }

  $("textCursor").classList.add("hidden");

  lastResultText=text;
  $("copyResultBtn").disabled=!text;

  if(result.status==="ACCEPT"&&result.canonical_text){
    ttsText=result.canonical_text;
    $("speakBtn").disabled=!supportsTts();
    $("ttsHint").textContent=
      "规范中文已生成，可播放标准普通话";

    $("checkText").className="done";
    $("checkText").querySelector("b").textContent="已生成";
  }else if(result.status==="CONFIRM"){
    ttsText="";
    $("speakBtn").disabled=true;
    $("ttsHint").textContent=
      "当前结果建议确认后再进行标准语音输出";

    $("checkText").className="";
    $("checkText").querySelector("b").textContent="待确认";
  }else{
    ttsText="";
    $("speakBtn").disabled=true;
    $("ttsHint").textContent=
      "未注册表达不生成错误标准语音";

    $("checkText").className="";
    $("checkText").querySelector("b").textContent="未生成";
  }
}

function showError(msg){
  $("decisionWaiting").classList.add("hidden");
  $("decisionResult").classList.remove("hidden");
  $("resultBadge").textContent="服务异常";
  $("canonicalText").textContent=msg;
  $("latencyText").textContent="模型实际耗时 -- ms";
}

/* -----------------------------
   TTS
----------------------------- */

function supportsTts(){
  return "speechSynthesis" in window &&
    "SpeechSynthesisUtterance" in window;
}

$("speakBtn").addEventListener("click",speak);
$("stopSpeakBtn").addEventListener("click",stopTts);

function speak(){
  if(!supportsTts()||!ttsText)return;

  window.speechSynthesis.cancel();

  const u=new SpeechSynthesisUtterance(ttsText);
  u.lang="zh-CN";
  u.rate=.94;
  u.pitch=1;

  const voices=window.speechSynthesis.getVoices();

  u.voice=
    voices.find(v=>/^zh-CN$/i.test(v.lang))||
    voices.find(v=>/^zh/i.test(v.lang))||
    null;

  u.onstart=()=>{
    $("speakBtn").disabled=true;
    $("stopSpeakBtn").disabled=false;
    $("ttsHint").textContent="正在播放标准普通话";
    $("ttsBars").classList.remove("hidden");

    $("checkTts").className="active";
    $("checkTts").querySelector("b").textContent="播报中";

    setClosure(3,"active","播报中");
  };

  u.onend=()=>{
    $("speakBtn").disabled=false;
    $("stopSpeakBtn").disabled=true;
    $("ttsHint").textContent="标准语音播放完成";
    $("ttsBars").classList.add("hidden");

    $("checkTts").className="done";
    $("checkTts").querySelector("b").textContent="已生成";

    setClosure(3,"done","完成");
  };

  u.onerror=()=>{
    $("speakBtn").disabled=false;
    $("stopSpeakBtn").disabled=true;
    $("ttsHint").textContent=
      "TTS 播放失败，请检查系统中文语音";
    $("ttsBars").classList.add("hidden");
  };

  window.speechSynthesis.speak(u);
}

function stopTts(){
  if(supportsTts()){
    window.speechSynthesis.cancel();
  }

  $("speakBtn").disabled=!ttsText;
  $("stopSpeakBtn").disabled=true;
  $("ttsBars").classList.add("hidden");
}

/* -----------------------------
   Audio analysis
----------------------------- */

async function decodeAudio(url){
  const res=await fetch(url);

  if(!res.ok){
    throw new Error("音频读取失败");
  }

  const AC=
    window.AudioContext||
    window.webkitAudioContext;

  if(!AC){
    throw new Error("浏览器不支持 WebAudio");
  }

  const ac=new AC();

  const buffer=await ac.decodeAudioData(
    (await res.arrayBuffer()).slice(0)
  );

  if(ac.close)await ac.close();

  return buffer;
}

function calculateAudioStats(buffer){
  const data=buffer.getChannelData(0);

  let sumSq=0;
  let peak=0;

  for(let i=0;i<data.length;i++){
    const v=Math.abs(data[i]);
    peak=Math.max(peak,v);
    sumSq+=data[i]*data[i];
  }

  const rms=Math.sqrt(
    sumSq/Math.max(1,data.length)
  );

  return{
    duration:buffer.duration,
    sampleRate:buffer.sampleRate,
    peak,
    rms,
    peakDb:20*Math.log10(Math.max(peak,1e-9)),
    rmsDb:20*Math.log10(Math.max(rms,1e-9))
  };
}

function renderAudioStats(stats){
  $("audioDuration").textContent=
    `${stats.duration.toFixed(2)} s`;

  $("sampleRate").textContent=
    `${stats.sampleRate} Hz`;

  $("peakDb").textContent=
    `${stats.peakDb.toFixed(1)} dBFS`;

  $("rmsDb").textContent=
    `${stats.rmsDb.toFixed(1)} dBFS`;
}

function clearAudioStats(){
  $("audioDuration").textContent="--";
  $("sampleRate").textContent="--";
  $("peakDb").textContent="--";
  $("rmsDb").textContent="--";
}

function fft(re,im){
  const n=re.length;

  for(let i=1,j=0;i<n;i++){
    let bit=n>>1;

    for(;j&bit;bit>>=1){
      j^=bit;
    }

    j^=bit;

    if(i<j){
      [re[i],re[j]]=[re[j],re[i]];
      [im[i],im[j]]=[im[j],im[i]];
    }
  }

  for(let len=2;len<=n;len<<=1){
    const ang=-2*Math.PI/len;
    const wlr=Math.cos(ang);
    const wli=Math.sin(ang);

    for(let i=0;i<n;i+=len){
      let wr=1;
      let wi=0;

      for(let j=0;j<len/2;j++){
        const ur=re[i+j];
        const ui=im[i+j];

        const vr=
          re[i+j+len/2]*wr-
          im[i+j+len/2]*wi;

        const vi=
          re[i+j+len/2]*wi+
          im[i+j+len/2]*wr;

        re[i+j]=ur+vr;
        im[i+j]=ui+vi;
        re[i+j+len/2]=ur-vr;
        im[i+j+len/2]=ui-vi;

        const nwr=wr*wlr-wi*wli;
        wi=wr*wli+wi*wlr;
        wr=nwr;
      }
    }
  }
}

function buildSpectrogram(buffer){
  const data=buffer.getChannelData(0);
  const N=256;
  const bins=128;
  const maxFrames=250;

  const hop=Math.max(
    64,
    Math.floor(
      Math.max(1,data.length-N)/maxFrames
    )
  );

  const frames=[];
  const energy=[];
  const linearMagnitude=[];

  for(
    let start=0;
    start+N<=data.length &&
    frames.length<maxFrames;
    start+=hop
  ){
    const re=new Array(N);
    const im=new Array(N).fill(0);

    let frameSq=0;

    for(let i=0;i<N;i++){
      const sample=data[start+i];

      frameSq+=sample*sample;

      re[i]=sample*
        (.5-.5*Math.cos(
          2*Math.PI*i/(N-1)
        ));
    }

    fft(re,im);

    const f=[];
    const magFrame=[];

    for(let k=0;k<bins;k++){
      const mag=Math.hypot(re[k],im[k]);

      magFrame.push(mag);

      f.push(
        20*Math.log10(mag+1e-6)
      );
    }

    frames.push(f);
    linearMagnitude.push(magFrame);

    const frameRms=
      Math.sqrt(frameSq/N);

    energy.push(
      20*Math.log10(
        Math.max(frameRms,1e-9)
      )
    );
  }

  let mn=Infinity;
  let mx=-Infinity;

  frames.forEach(f=>
    f.forEach(v=>{
      mn=Math.min(mn,v);
      mx=Math.max(mx,v);
    })
  );

  return{
    frames,
    linearMagnitude,
    energy,
    bins,
    mn,
    mx,
    window:N,
    sampleRate:buffer.sampleRate
  };
}

function spectrumColor(t){
  t=Math.max(0,Math.min(1,t));

  if(t<.3){
    const u=t/.3;
    return `rgb(${4+Math.round(8*u)},${17+Math.round(40*u)},${27+Math.round(70*u)})`;
  }

  if(t<.66){
    const u=(t-.3)/.36;
    return `rgb(${12+Math.round(25*u)},${57+Math.round(88*u)},${97+Math.round(70*u)})`;
  }

  const u=(t-.66)/.34;

  return `rgb(${37+Math.round(115*u)},${145+Math.round(88*u)},${167+Math.round(65*u)})`;
}

function spectrogramReveal(progress){
  if(progress<.14){
    return progress*.35;
  }

  if(progress<.72){
    const local=(progress-.14)/.58;
    return .049+local*.73;
  }

  const local=(progress-.72)/.28;
  return .779+local*.221;
}

function currentSpectrumProgress(){
  const raw=parseInt(
    $("progressText").textContent,
    10
  )||0;

  return spectrogramReveal(
    Math.max(0,Math.min(1,raw/100))
  );
}

function drawSpectrogram(progress){
  const ctx=specCanvas.getContext("2d");
  const r=specCanvas.getBoundingClientRect();
  const dpr=Math.max(
    1,
    window.devicePixelRatio||1
  );

  const w=Math.max(
    1,
    Math.floor(r.width*dpr)
  );

  const h=Math.max(
    1,
    Math.floor(r.height*dpr)
  );

  if(
    specCanvas.width!==w ||
    specCanvas.height!==h
  ){
    specCanvas.width=w;
    specCanvas.height=h;
  }

  ctx.fillStyle="#040d14";
  ctx.fillRect(0,0,w,h);

  if(
    !specData ||
    !specData.frames.length
  ){
    return;
  }

  const count=Math.floor(
    specData.frames.length*progress
  );

  const frameW=
    w/specData.frames.length;

  const binH=
    h/specData.bins;

  const range=Math.max(
    1e-6,
    specData.mx-specData.mn
  );

  for(let x=0;x<count;x++){
    const frame=specData.frames[x];

    for(let k=0;k<specData.bins;k++){
      let t=
        (frame[k]-specData.mn)/range;

      t=Math.pow(t,1.65);

      ctx.fillStyle=
        spectrumColor(t);

      ctx.fillRect(
        x*frameW,
        h-(k+1)*binH,
        Math.ceil(frameW+1),
        Math.ceil(binH+1)
      );
    }
  }
}

function drawFrequencySpectrum(progress){
  const ctx=freqCanvas.getContext("2d");
  const r=freqCanvas.getBoundingClientRect();
  const dpr=Math.max(
    1,
    window.devicePixelRatio||1
  );

  const w=Math.max(
    1,
    Math.floor(r.width*dpr)
  );

  const h=Math.max(
    1,
    Math.floor(r.height*dpr)
  );

  if(
    freqCanvas.width!==w ||
    freqCanvas.height!==h
  ){
    freqCanvas.width=w;
    freqCanvas.height=h;
  }

  ctx.fillStyle="#040d14";
  ctx.fillRect(0,0,w,h);

  if(
    !specData ||
    !specData.linearMagnitude.length
  ){
    return;
  }

  const frameIndex=Math.min(
    specData.linearMagnitude.length-1,
    Math.max(
      0,
      Math.floor(
        specData.linearMagnitude.length*progress
      )
    )
  );

  const frame=
    specData.linearMagnitude[frameIndex];

  const maxMag=Math.max(
    1e-9,
    ...frame
  );

  const values=
    frame.map(v=>
      Math.pow(v/maxMag,.35)
    );

  const barW=w/values.length;

  // filled spectrum
  for(let i=0;i<values.length;i++){
    const v=values[i];
    const barH=v*h*.83;

    const grad=ctx.createLinearGradient(
      0,
      h-barH,
      0,
      h
    );

    grad.addColorStop(0,"rgba(125,220,245,.95)");
    grad.addColorStop(.5,"rgba(60,166,211,.72)");
    grad.addColorStop(1,"rgba(24,91,125,.35)");

    ctx.fillStyle=grad;

    ctx.fillRect(
      i*barW,
      h-barH,
      Math.max(1,barW-1),
      barH
    );
  }

  let maxIndex=0;

  for(let i=1;i<frame.length;i++){
    if(frame[i]>frame[maxIndex]){
      maxIndex=i;
    }
  }

  const freq=
    maxIndex*
    specData.sampleRate/
    specData.window;

  $("dominantFreq").textContent=
    `${Math.round(freq)} Hz`;

  $("freqReadout").textContent=
    `${Math.round(freq)} Hz`;
}

function updateLiveReadout(progress){
  if(
    !specData ||
    !specData.frames.length
  ){
    return;
  }

  const index=Math.min(
    specData.frames.length-1,
    Math.max(
      0,
      Math.floor(
        specData.frames.length*progress
      )
    )
  );

  $("frameReadout").textContent=
    `${index+1} / ${specData.frames.length}`;

  const energy=
    specData.energy[index];

  $("energyReadout").textContent=
    Number.isFinite(energy)
    ?`${energy.toFixed(1)} dB`
    :"-- dB";
}

function resetSpectrogram(){
  const ctx=specCanvas.getContext("2d");
  const r=specCanvas.getBoundingClientRect();
  const dpr=Math.max(
    1,
    window.devicePixelRatio||1
  );

  specCanvas.width=Math.max(
    1,
    Math.floor(r.width*dpr)
  );

  specCanvas.height=Math.max(
    1,
    Math.floor(r.height*dpr)
  );

  ctx.fillStyle="#040d14";
  ctx.fillRect(
    0,0,
    specCanvas.width,
    specCanvas.height
  );

  specIdle.classList.remove("hidden");
  specScan.classList.add("hidden");
}

function resetFrequencySpectrum(){
  const ctx=freqCanvas.getContext("2d");
  const r=freqCanvas.getBoundingClientRect();
  const dpr=Math.max(
    1,
    window.devicePixelRatio||1
  );

  freqCanvas.width=Math.max(
    1,
    Math.floor(r.width*dpr)
  );

  freqCanvas.height=Math.max(
    1,
    Math.floor(r.height*dpr)
  );

  ctx.fillStyle="#040d14";
  ctx.fillRect(
    0,0,
    freqCanvas.width,
    freqCanvas.height
  );
}

window.addEventListener(
  "resize",
  ()=>{
    const p=currentSpectrumProgress();
    drawSpectrogram(p);
    drawFrequencySpectrum(p);
  }
);

boot();
