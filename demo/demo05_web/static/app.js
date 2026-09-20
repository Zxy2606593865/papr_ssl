/* Voice Bridge v5 UI-locked integration; continuous phrase spotting is isolated to the original second-session structure. */
"use strict";
const $ = id => document.getElementById(id);
const state = {presets:[],profile:null,health:null,mock:false,mode:"instant",session:"single",cache:new Map(),continuousResults:[],buffer:null,stats:null,spectrum:null,wave:[],loading:false,running:false,run:0,load:0,controller:null,tts:"",utterance:0,history:[],paused:false,progress:0,frameProgress:0};
const PRESENTATION_MS = 30000;
const CONTINUOUS_API = "/api/continuous/recognize-demo";
// Demo-05 current continuous backend contract.
// The backend returns detections[] with start_sec/end_sec; normalize it here so
// the integrated v5 UI can keep one internal timeline shape.
async function continuousApi(item,signal){
  if(!item || item.id!=="continuous_natural_30s") throw new Error("当前后端只注册了 continuous_natural_30s 长语音演示样本。");
  const raw=await api(CONTINUOUS_API,{method:"POST",signal},120000);
  if(!raw || !Array.isArray(raw.detections)) throw new Error("长语音接口响应缺少 detections。");
  return {
    ...raw,
    segments:raw.detections.map(d=>({
      start:Number(d.start_sec),
      end:Number(d.end_sec),
      status:String(d.status||""),
      canonical_text:typeof d.canonical_text==="string"?d.canonical_text:"",
      merged_atomic_spans:d.merged_atomic_spans
    }))
  };
}
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const EXPLANATIONS = [
  ["读取声音", "流程说明：从已录制的语音样本开始，输入并非自由对话。"],
  ["检查音频", "流程说明：采样率、时长和能量帮助检查音频质量。"],
  ["查看频率分布", "流程说明：帧频谱来自输入音频计算，不是模型内部特征。"],
  ["理解个性化发音", "流程说明：个性化识别依赖该学生已准备的语音样本。"],
  ["关联已注册表达", "流程说明：在约定的表达范围内匹配，不任意补全语义。"],
  ["检查输出边界", "流程说明：本接口只返回最终决策，不提供内部逐步推理。"],
  ["保留不确定性", "流程说明：待确认或未识别的声音，不自动播报为确定答案。"],
  ["连接清晰表达", "流程说明：成功匹配后，规范中文可交给普通话语音合成。"]
];
const CONTINUOUS_EXPLANATIONS = [
  ["读取长语音", "流程说明：整段连续语音一次提交，不在浏览器内预先切成单句。"],
  ["检查整段音频", "流程说明：采样率、总时长和能量帮助检查长语音质量。"],
  ["查看频率分布", "流程说明：帧频谱来自输入音频计算，不是模型内部特征。"],
  ["寻找候选边界", "流程说明：Continuous Phrase Spotter 在整段音频中定位候选时间范围。"],
  ["关联已注册表达", "流程说明：只在已注册表达范围内匹配，不执行开放式长文本转写。"],
  ["整理时间片段", "流程说明：返回片段按起止时间排序，并检查越界与异常重叠。"],
  ["保留不确定性", "流程说明：待确认或未匹配片段不会被改写成确定结果。"],
  ["输出识别时间轴", "流程说明：全部片段确认后，规范中文才能合并用于普通话播报。"]
];
const wait = ms => new Promise(resolve => setTimeout(resolve,ms));
const show = (id,visible=true) => $(id).classList.toggle("hidden",!visible);
const set = (id,text) => {$(id).textContent=text;};
const supportsTts = () => "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
const safeCount = value => Number.isInteger(value) && value >= 0 ? value : null;
function setRecognizeLabel(label){const b=$("recognizeBtn");b.replaceChildren(document.createTextNode(label+" "));const arrow=document.createElement("span");arrow.textContent="↗";b.append(arrow);}

async function api(url,options={},timeoutMs=25000) {
  const controller = new AbortController();
  const upstream = options.signal;
  const abort = () => controller.abort();
  upstream?.addEventListener("abort",abort,{once:true});
  if(upstream?.aborted) controller.abort();
  let timeout=false;
  const timer=setTimeout(()=>{timeout=true;controller.abort();},timeoutMs);
  try {
    const response=await fetch(url,{...options,signal:controller.signal,cache:"no-store"});
    if(!response.ok) throw new Error(`服务返回 ${response.status}，请检查后端连接。`);
    return await response.json();
  } catch(error) {
    if(timeout) throw new Error("接口等待超时，请检查服务后重试。");
    throw error;
  } finally {clearTimeout(timer);upstream?.removeEventListener("abort",abort);}
}

function setMock(isMock) {
  state.mock=Boolean(isMock);
  show("demoNotice",state.mock);
  set("footerSource",state.mock?"MOCK 预览 · 固定接口结果 · 非模型推理":"服务端接口 · 音频可视化 · 普通话播报");
  set("guideSource",state.mock?"当前由本地 Mock 提供固定结果与测试声音。接入合作伙伴模型后，才能验证真实识别。":"模型与学生音频由接入的后端提供；界面本身不执行模型推理。");
}

function populatePresetSelect(){
  const select=$("presetSelect"),continuous=state.session==="sequence";
  const allowed=state.presets.filter(p=>continuous?p.demo_kind==="continuous":p.demo_kind!=="continuous");
  select.replaceChildren(new Option(continuous?"请选择一段长语音":"请选择一条语音样本",""));
  if(continuous){
    const group=document.createElement("optgroup");group.label="长语音 / 连续检测";
    allowed.forEach(p=>group.append(new Option(p.display_name||p.id,p.id)));
    if(group.children.length)select.append(group);
  }else{
    for(const [kind,label] of [["recognition","个性化识别"],["confirmation","人工确认"],["rejection","未知表达拒识"]]){
      const group=document.createElement("optgroup");group.label=label;
      allowed.filter(p=>p.demo_kind===kind).forEach(p=>group.append(new Option(p.display_name||p.id,p.id)));
      if(group.children.length)select.append(group);
    }
    allowed.filter(p=>!["recognition","confirmation","rejection"].includes(p.demo_kind)).forEach(p=>select.append(new Option(p.display_name||p.id,p.id)));
  }
  select.disabled=!allowed.length;
  if(allowed.length)select.value=allowed[0].id;
  return allowed.length;
}

async function boot() {
  show("connectionError",false);
  set("systemStatus","正在连接服务");
  $("retryBootBtn").disabled=true;
  $("presetSelect").disabled=true;
  try {
    const [health,profile,collection]=await Promise.all([api("/api/health"),api("/api/profile"),api("/api/presets")]);
    if(health.ok===false || !Array.isArray(collection.presets)) throw new Error("服务尚未就绪或样本格式不正确。");
    state.health=health;state.profile=profile;
    state.presets=collection.presets.filter(p=>typeof p.id==="string" && typeof p.audio_url==="string");
    setMock(health.demo_mode===true || state.presets.some(p=>p.audio_url.startsWith("/mock/")));
    set("systemStatus",state.mock?"模拟服务已连接":"服务已连接");
    set("modelState",state.mock?"MOCK · 无真实模型":health.model_loaded?"模型已加载":"模型未就绪");
    set("studentName",profile.student_name||"未命名档案");
    const n=safeCount(profile.registered_expression_count), each=safeCount(profile.enrollment_per_expression);
    for(const id of ["taskRegistered","registeredCount"]) set(id,n??"—");
    for(const id of ["taskSamples","sampleCount"]) set(id,n!==null && each!==null?n*each:"—");
    const available=populatePresetSelect();
    if(!available){set("sampleDescription","暂无可用样本，请让后端提供演示音频。");set("taskStatus","暂无样本");}
    else await selectPreset();
  } catch(error) {
    state.health=null;
    set("systemStatus","服务未连接");set("modelState","无法读取模型状态");
    set("connectionMessage",`连接失败：${error.message}`);show("connectionError");
    $("presetSelect").replaceChildren(new Option("未能加载样本",""));
    syncControls();
  } finally {$("retryBootBtn").disabled=false;}
}

function syncControls() {
  const selected=Boolean($("presetSelect").value);
  const canRun=Boolean(state.buffer&&selected&&state.health?.model_loaded&&!state.loading&&!state.running);
  $("recognizeBtn").disabled=!canRun;$("restartBtn").disabled=!canRun;
  $("presetSelect").disabled=state.running||!state.presets.length;
  $("playOriginalBtn").disabled=!state.buffer||state.loading;
  show("cancelBtn",state.running);show("restartBtn",!state.running);
  document.querySelectorAll(".mode-btn").forEach(b=>b.disabled=state.running);
  document.querySelectorAll(".session-tab").forEach(b=>b.disabled=state.running||state.loading);
  $("addQueueBtn").disabled=state.running||state.loading||!selected;
  $("fillQueueBtn").disabled=state.running||state.loading||!state.presets.some(p=>p.demo_kind==="continuous");
  $("clearQueueBtn").disabled=state.running||state.loading;
  document.querySelectorAll("#queueList button").forEach(b=>b.disabled=state.running||state.loading);
  setRecognizeLabel(state.session==="sequence"?"开始长语音识别":"开始识别");
}
function setClosure(i,cls,label) {
  const item=$("closureFlow").querySelector(`[data-step="${i}"]`);
  item.className=cls;item.querySelector("b").textContent=label;
}
function setProgress(progress) {
  state.progress=progress;
  set("progressText",`${Math.round(progress*100)}%`);
  $("progressFill").style.width=`${progress*100}%`;
}
function clearResult() {
  stopTts();state.tts="";
  state.continuousResults=[];$("sequenceResults").replaceChildren();show("sequenceResults",false);
  show("decisionWaiting");show("decisionResult",false);
  $("resultCard").removeAttribute("data-result");
  set("canonicalText","");set("decisionBadge","待识别");set("copyResultBtn","复制识别结果 ↗");
  set("ttsHint","仅在全部相关结果确认成功后提供普通话播报。");
  $("speakBtn").disabled=true;$("copyResultBtn").disabled=true;
  for(const [id,label] of [["checkText","待生成"],["checkTts","待播报"]]) {$(id).className="";$(id).querySelector("b").textContent=label;}
  for(let i=0;i<5;i++) setClosure(i,"",i===4?"未接入":i===0?"就绪":"等待");
}
function resetTask() {
  clearResult();document.body.dataset.state="idle";
  set("heroTitle","等待开始识别");
  set("heroCaption","播放语音，查看接口返回的识别结果。");
  set("currentStage","等待语音接入");set("analysisBadge","准备就绪");set("streamState","就绪");
  [...$("stageTrack").children].forEach(e=>e.className="");
  setProgress(0);clearEvents();
  const now=new Date();set("taskId",`VB-${String(now.getMonth()+1).padStart(2,"0")}${String(now.getDate()).padStart(2,"0")}-${String(now.getTime()).slice(-4)}`);
}

async function presentUntilOutcome(token,started,getOutcome,explanations,presentationPrefix,instantStage) {
  let nextStage=0;
  while(token===state.run) {
    const outcome=getOutcome();
    if(outcome?.error) throw outcome.error;
    const elapsed=performance.now()-started;
    if(state.mode==="presentation") {
      const progress=Math.min(elapsed/PRESENTATION_MS,1);setProgress(progress);
      while(nextStage<explanations.length && elapsed>=nextStage*PRESENTATION_MS/explanations.length) {
        [...$("stageTrack").children].forEach((e,i)=>e.className=i<nextStage?"done":i===nextStage?"active":"");
        set("currentStage",`${presentationPrefix} · ${explanations[nextStage][0]}`);
        addEvent(explanations[nextStage][1],"讲解示意","info");nextStage++;
      }
      if(progress>=1 && outcome) return;
    } else {
      set("currentStage",instantStage);set("progressText","—");
      if(outcome) return;
    }
    await wait(80);
  }
}

async function selectPreset() {
  const version=++state.load;++state.run;state.controller?.abort();state.running=false;
  $("audioPlayer").pause();set("playOriginalBtn","试听原始语音");resetTask();
  state.buffer=null;state.stats=null;state.spectrum=null;state.wave=[];state.frameProgress=0;
  for(const id of ["audioDuration","sampleRate","peakDb","rmsDb"]) set(id,"—");
  for(const [id,value] of [["freqReadout","— Hz"],["nyquistLabel","— kHz"]]) set(id,value);
  renderVisuals(0);
  const item=state.presets.find(p=>p.id===$("presetSelect").value);
  if(!item){$("audioPlayer").removeAttribute("src");$("audioPlayer").load();set("taskStatus","等待接入");set("taskBadge","待接入");state.loading=false;syncControls();return;}
  state.loading=true;syncControls();set("taskStatus","载入音频");set("taskBadge","载入中");
  set("sampleDescription",item.description||"后端提供的语音样本");
  try {
    // Do not contact arbitrary remote hosts for student audio from untrusted profile data.
    const audioUrl=new URL(item.audio_url,location.origin);
    if(audioUrl.origin!==location.origin) throw new Error("音频需通过同源后端提供，请检查 audio_url。");
    $("audioPlayer").src=audioUrl.href;$("audioPlayer").load();
    const response=await fetch(audioUrl.href);
    if(!response.ok) throw new Error("音频文件读取失败。");
    const bytes=await response.arrayBuffer();
    if(version!==state.load) return;
    const AC=window.AudioContext||window.webkitAudioContext;
    if(!AC) throw new Error("浏览器不支持 Web Audio，请使用新版浏览器。");
    const context=new AC();let buffer;
    try{buffer=await context.decodeAudioData(bytes);}finally{await context.close();}
    if(version!==state.load) return;
    state.cache.set(item.id,buffer);state.buffer=buffer;state.stats=audioStats(buffer);state.spectrum=makeSpectrum(buffer);state.wave=makeWave(buffer);
    set("audioDuration",`${buffer.duration.toFixed(2)} s`);set("sampleRate",`${buffer.sampleRate} Hz`);
    set("peakDb",`${state.stats.peakDb.toFixed(1)} dBFS`);set("rmsDb",`${state.stats.rmsDb.toFixed(1)} dBFS`);
    set("nyquistLabel",`${(buffer.sampleRate/2000).toFixed(1)} kHz`);
    set("taskStatus",state.health.model_loaded?"已就绪":"模型未就绪");set("taskBadge","已接入");
    set("currentStage",state.health.model_loaded?"声音已就绪，等待开始":"等待后端模型就绪");
    set("waveLabel",state.mock?"测试音频 · 振幅波形":"所选音频 · 振幅波形");
    renderVisuals(0);setClosure(0,"done","已载入");
    addEvent(`音频已载入：${buffer.duration.toFixed(2)} 秒 / 解码 ${buffer.sampleRate} Hz`,"已计算");
  } catch(error) {
    if(version!==state.load) return;
    state.buffer=null;state.stats=null;
    set("taskStatus","音频不可用");set("taskBadge","载入失败");set("sampleDescription",error.message);
    set("currentStage","请重新选择或检查音频文件");addEvent(error.message,"音频异常","warn");
  } finally {if(version===state.load){state.loading=false;renderContinuousBuilder();syncControls();}}
}

async function recognize() {
  if($("recognizeBtn").disabled) return;
  if(state.session==="sequence")return recognizeContinuous();
  $("audioPlayer").pause();clearResult();clearEvents();
  $("audioPlayer").currentTime=0;$("audioPlayer").play().catch(()=>addEvent("原始声音未能自动播放，可手动试听。","提示","warn"));
  [...$("stageTrack").children].forEach(e=>e.className="");
  const token=++state.run,controller=new AbortController();state.controller=controller;state.running=true;
  document.body.dataset.state="running";$("resultCard").setAttribute("aria-busy","true");syncControls();
  set("taskStatus","处理中");set("taskBadge","处理中");set("decisionBadge","处理中");set("analysisBadge",state.mode==="presentation"?"讲解进行中":"正在识别");set("streamState","记录中");
  set("heroTitle","识别请求已发送");set("heroCaption",state.mode==="presentation"?"接口即时请求；下方流程说明按讲解节奏展开。":"正在等待服务端返回，不估算模型内部进度。");
  setClosure(0,"done","已接入");setClosure(1,"active","请求中");
  setProgress(0);
  addEvent(state.mock?"已提交模拟识别请求。":"已向后端提交识别请求。","已发送");
  let outcome=null;
  const started=performance.now();
  // Start immediately: presentation duration is never reported as model latency.
  const pending=api(`/api/recognize/${encodeURIComponent($("presetSelect").value)}`,{method:"POST",signal:controller.signal})
    .then(result=>{outcome={result};if(token===state.run){setClosure(1,"done","已返回");addEvent(state.mock?"模拟接口已返回固定结果。":"服务端已返回结果。","已返回");}})
    .catch(error=>{outcome={error};});
  try {
    await presentUntilOutcome(token,started,()=>outcome,EXPLANATIONS,"流程讲解","后端识别请求进行中");
    await pending;
    if(token!==state.run) return;
    const result=outcome.result;
    validateResult(result);
    setProgress(1);renderVisuals(1);
    if(state.mode==="presentation") [...$("stageTrack").children].forEach(e=>e.className="done");
    set("currentStage",state.mode==="presentation"?"流程讲解结束 · 结果已呈现":"识别完成 · 结果已呈现");
    set("analysisBadge","已完成");set("taskStatus","已完成");set("taskBadge","已完成");set("streamState","已完成");
    revealResult(result);
    if(result.status==="ACCEPT" && $("autoTts").checked) speak();
  } catch(error) {
    if(token!==state.run) return;
    showError(error.message);addEvent(error.message,"异常","warn");
  } finally {
    if(token===state.run){state.running=false;$("resultCard").removeAttribute("aria-busy");syncControls();}
  }
}

function cancelRun() {
  if(!state.running)return;
  ++state.run;state.controller?.abort();state.running=false;stopTts();state.tts="";$("speakBtn").disabled=true;$("copyResultBtn").disabled=true;$("audioPlayer").pause();$("audioPlayer").controls=true;document.body.dataset.state="idle";
  $("resultCard").removeAttribute("aria-busy");
  if(state.session==='sequence'&&state.continuousResults.length){
    set('canonicalText','长语音识别已取消');set('resultBadge','已取消 · 已返回结果不再继续处理');set('ttsHint','本次未完整结束，已阻止合并播报。');
  }
  set("currentStage","本次已取消，可以重新开始");set("analysisBadge","已取消");set("decisionBadge","已取消");set("taskBadge","已取消");set("taskStatus","已取消");set("streamState","已取消");
  set("heroTitle","本次已取消");set("heroCaption","已停止后续处理；可以重新开始。");setClosure(0,"","已停止");setClosure(1,"","已取消");setClosure(2,"","已取消");setClosure(3,"","未播报");addEvent("本次展示已取消，忽略其后到达的结果。","已取消","warn");syncControls();
}

function revealResult(result) {
  const accepted=result.status==="ACCEPT",confirm=result.status==="CONFIRM";
  document.body.dataset.state=accepted?"accepted":confirm?"confirm":"rejected";
  $("resultCard").dataset.result=result.status;show("decisionWaiting",false);show("decisionResult");
  const status=accepted?"识别成功":confirm?"需要人工确认":"未识别到已注册表达";
  set("decisionBadge",accepted?"已识别":confirm?"待确认":"未匹配");set("resultBadge",state.mock?`${status} · 模拟`:status);
  // Never turn an uncertain candidate into a canonical assertion.
  const text=accepted?result.canonical_text:confirm?"需要人工确认":"未匹配已注册表达";
  set("outputLabel",accepted?"规范中文表达":confirm?"需要人工确认":"未匹配已注册表达");set("canonicalText",text);
  const latency=result.latency_ms;
  set("latencyText",typeof latency==="number"&&Number.isFinite(latency)&&latency>=0?`${state.mock?"模拟耗时":"后端报告耗时"} ${latency.toFixed(1)} ms`:`${state.mock?"模拟接口":"后端"}未提供耗时`);
  state.tts=accepted?result.canonical_text:"";$("speakBtn").disabled=!accepted||!supportsTts();$("copyResultBtn").disabled=!accepted;
  set("ttsHint",accepted?(supportsTts()?"规范中文已生成，可播放标准普通话。":"当前浏览器不支持语音合成，可复制文字。"):confirm?"请由熟悉学生的教师确认含义，暂不自动播报。":"请重新输入或由教师确认，暂不输出普通话。");
  $("checkText").className=accepted?"done":"";$("checkText").querySelector("b").textContent=accepted?"已生成":confirm?"待确认":"未生成";
  setClosure(1,"done","已返回");setClosure(2,accepted?"done":"",accepted?"已生成":confirm?"待确认":"未匹配");
  set("heroTitle",accepted?"识别完成":confirm?"结果需要确认":"本次未匹配");
  set("heroCaption",accepted?"接口已返回规范中文，可播放普通话。":"未产生可直接播报的确定结果，请由教师确认。");
  addEvent(status,accepted?"成功":"需关注",accepted?"ok":"warn");
}
function showError(message) {
  stopTts();state.tts="";document.body.dataset.state="error";
  $("resultCard").dataset.result="ERROR";show("decisionWaiting",false);show("decisionResult");
  set("decisionBadge","服务异常");set("resultBadge","请求未完成");set("outputLabel","本次未产生识别结果");set("canonicalText","暂时无法识别");set("ttsHint",message);set("latencyText","本次无有效模型耗时");
  set("currentStage","请求未完成，请检查服务后重试");set("heroTitle","识别请求未完成");set("heroCaption",message);set("analysisBadge","异常");set("taskBadge","异常");set("taskStatus","异常");set("streamState","异常");
  $("speakBtn").disabled=true;$("copyResultBtn").disabled=true;setClosure(1,"","异常");
}

function clearEvents(){state.history=[];$("eventStream").replaceChildren();}
function addEvent(text,status,cls="ok") {
  const time=new Date().toLocaleTimeString("zh-CN",{hour12:false});
  state.history.push({time,text,status});
  const row=document.createElement("div");row.className=`event entering ${cls}`;
  const timeNode=document.createElement("time"),message=document.createElement("span"),badge=document.createElement("b");
  timeNode.textContent=time;message.textContent=text;badge.textContent=status;row.append(timeNode,message,badge);$("eventStream").append(row);
  if(!state.paused)$("eventStream").scrollTop=$("eventStream").scrollHeight;
}
function exportEvents(){
  const lines=["声桥 · 本地演示记录",`模式：${state.mode} / ${state.mock?"MOCK（非真实推理）":"后端接口"}`,"流程讲解是示意，不是模型内部推理日志。",...state.history.map(e=>`${e.time}\t${e.status}\t${e.text}`)];
  const url=URL.createObjectURL(new Blob([lines.join("\n")],{type:"text/plain;charset=utf-8"}));const a=document.createElement("a");a.href=url;a.download=`${$("taskId").textContent}_trace.txt`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

function stopTts(){
  ++state.utterance;
  if(supportsTts())window.speechSynthesis.cancel();
  show("ttsBars",false);$("stopSpeakBtn").disabled=true;$("speakBtn").disabled=!state.tts||!supportsTts();
  if($("checkTts").className==="active"){$("checkTts").className="";$("checkTts").querySelector("b").textContent="已停止";set("ttsHint","播报已停止，可重新播放。");setClosure(3,"","已停止");}
}
function speak(){
  if(!state.tts||!supportsTts())return;
  $("audioPlayer").pause();
  stopTts();const token=state.utterance;const utterance=new SpeechSynthesisUtterance(state.tts);
  utterance.lang="zh-CN";utterance.rate=.94;
  // Prefer a local voice to avoid silently sending student-linked text to a remote speech service.
  const voices=speechSynthesis.getVoices();const local=voices.find(v=>v.localService&&/^zh[-_]CN$/i.test(v.lang))||voices.find(v=>v.localService&&/^zh/i.test(v.lang));
  if(!local){set("ttsHint","未检测到本机中文语音，请先安装系统中文语音后重试。");return;}
  utterance.voice=local;
  utterance.onstart=()=>{if(token!==state.utterance)return;$("speakBtn").disabled=true;$("stopSpeakBtn").disabled=false;set("ttsHint","正在播放普通话");show("ttsBars");$("checkTts").className="active";$("checkTts").querySelector("b").textContent="播报中";setClosure(3,"active","播报中");};
  utterance.onend=()=>{if(token!==state.utterance)return;show("ttsBars",false);$("speakBtn").disabled=false;$("stopSpeakBtn").disabled=true;set("ttsHint","普通话播报完成");$("checkTts").className="done";$("checkTts").querySelector("b").textContent="已播报";setClosure(3,"done","已播报");};
  utterance.onerror=()=>{if(token!==state.utterance)return;show("ttsBars",false);$("speakBtn").disabled=false;$("stopSpeakBtn").disabled=true;set("ttsHint","播报未完成，请检查本机中文语音后重试。");$("checkTts").className="";$("checkTts").querySelector("b").textContent="未完成";setClosure(3,"","未完成");};
  speechSynthesis.speak(utterance);
}

/* DSP is visualization of the decoded input, not a claim about the model's features. */
function audioStats(buffer){let sum=0,peak=0;const data=buffer.getChannelData(0);for(const v of data){sum+=v*v;peak=Math.max(peak,Math.abs(v));}return {peakDb:20*Math.log10(Math.max(peak,1e-9)),rmsDb:20*Math.log10(Math.max(Math.sqrt(sum/Math.max(1,data.length)),1e-9))};}
function makeWave(buffer){const data=buffer.getChannelData(0);return Array.from({length:200},(_,i)=>{const a=Math.floor(i*data.length/200),b=Math.max(a+1,Math.floor((i+1)*data.length/200));let peak=0;for(let j=a;j<Math.min(b,data.length);j++)peak=Math.max(peak,Math.abs(data[j]));return peak;});}
function fft(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}for(let len=2;len<=n;len<<=1){const ang=-2*Math.PI/len,wr0=Math.cos(ang),wi0=Math.sin(ang);for(let i=0;i<n;i+=len){let wr=1,wi=0;for(let j=0;j<len/2;j++){const a=i+j,b=a+len/2,vr=re[b]*wr-im[b]*wi,vi=re[b]*wi+im[b]*wr,ur=re[a],ui=im[a];re[a]=ur+vr;im[a]=ui+vi;re[b]=ur-vr;im[b]=ui-vi;const next=wr*wr0-wi*wi0;wi=wr*wi0+wi*wr0;wr=next;}}}}
function makeSpectrum(buffer){const data=buffer.getChannelData(0),N=256,bins=128,count=Math.min(250,Math.max(1,Math.ceil(data.length/N))),frames=[],energy=[];let min=Infinity,max=-Infinity;
  for(let t=0;t<count;t++){const start=Math.floor(t*Math.max(0,data.length-N)/Math.max(1,count-1)),re=new Array(N),im=new Array(N).fill(0);let sq=0;for(let j=0;j<N;j++){const s=data[start+j]||0;sq+=s*s;re[j]=s*(.5-.5*Math.cos(2*Math.PI*j/(N-1)));}fft(re,im);const mags=re.slice(0,bins).map((v,k)=>Math.hypot(v,im[k]));const db=mags.map(v=>20*Math.log10(v+1e-6));for(const v of db){min=Math.min(min,v);max=Math.max(max,v);}frames.push({mags,db});energy.push(20*Math.log10(Math.max(Math.sqrt(sq/N),1e-9)));}
  return {frames,energy,min,max,bins,N,sampleRate:buffer.sampleRate};}
function canvasContext(id){const canvas=$(id),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return {ctx:canvas.getContext("2d"),w,h,dpr};}
function drawWave(progress){const {ctx,w,h,dpr}=canvasContext("waveCanvas");ctx.clearRect(0,0,w,h);ctx.fillStyle="#a9ccde";if(!state.wave.length){ctx.fillRect(0,h/2,w,1);set("waveTime","0 / —");return;}const peak=Math.max(.01,...state.wave),bar=w/state.wave.length;state.wave.forEach((v,i)=>{const height=Math.max(2*dpr,v/peak*h*.85);ctx.fillStyle=i/state.wave.length<=progress?"#24a6b1":"#83b5d4";ctx.fillRect(i*bar,h/2-height/2,Math.max(dpr,bar*.45),height);});set("waveTime",`${(progress*state.buffer.duration).toFixed(1)}s / ${state.buffer.duration.toFixed(1)}s`);}
function renderVisuals(progress){progress=Math.max(0,Math.min(1,Number.isFinite(progress)?progress:0));state.frameProgress=progress;drawWave(progress);const s=state.spectrum;
  const f=canvasContext("freqCanvas");f.ctx.fillStyle="#f1f8fd";f.ctx.fillRect(0,0,f.w,f.h);if(!s)return;
  const index=Math.min(s.frames.length-1,Math.floor(progress*s.frames.length)),frame=s.frames[index].mags,max=Math.max(1e-9,...frame),width=f.w/frame.length;const gradient=f.ctx.createLinearGradient(0,0,0,f.h);gradient.addColorStop(0,"#2a7ccf");gradient.addColorStop(1,"#51bfc2");f.ctx.fillStyle=gradient;
  frame.forEach((v,i)=>{const h=Math.pow(v/max,.4)*f.h*.86;f.ctx.fillRect(i*width,f.h-h,Math.max(1,width*.64),h);});
  const peak=frame.indexOf(Math.max(...frame)),hz=Math.round(peak*s.sampleRate/s.N);set("freqReadout",`${hz} Hz`);
}

function validateResult(result){
  if(!result||!["ACCEPT","CONFIRM","REJECT"].includes(result.status))throw new Error("接口返回了未知识别状态，已阻止语音播报。");
  if(result.status==="ACCEPT"&&(typeof result.canonical_text!=="string"||!result.canonical_text.trim()))throw new Error("识别成功响应缺少规范文本，已阻止语音播报。");
  return result;
}
function setMode(mode){
  state.mode=mode;document.body.dataset.mode=mode;
  document.querySelectorAll('.mode-btn').forEach(b=>{b.classList.toggle('active',b.dataset.mode===mode);b.setAttribute('aria-pressed',String(b.dataset.mode===mode));});
  set('modeHint',state.session==='sequence'?(mode==='presentation'?'长语音请求即时发出，流程说明按 30 秒讲解节奏展开':'长语音一次提交，识别结果按时间轴显示'):mode==='presentation'?'流程动画用于讲解，模型请求即时发出':'不附加展示等待，接口返回后立即显示结果');
  set('presentationNote',state.session==='sequence'?(mode==='presentation'?'长语音流程讲解不是模型内部思维；连续模式只识别已注册表达。':'连续模式识别已注册表达，不等同于开放式长文本 ASR。'):'流程讲解不是模型内部思维，实际耗时以接口返回为准。');
}
function renderContinuousRows(){
  const list=$('sequenceResults');list.replaceChildren();
  state.continuousResults.forEach((item,i)=>{
    const row=document.createElement('div');row.className='segment-result';row.dataset.status=item.status;
    const number=document.createElement('small'),time=document.createElement('span'),text=document.createElement('strong'),status=document.createElement('em');
    number.textContent=String(i+1).padStart(2,'0');time.textContent=`${item.start.toFixed(2)}–${item.end.toFixed(2)} s`;
    text.textContent=item.status==='ACCEPT'?item.canonical_text:item.status==='CONFIRM'?'需要人工确认':'未匹配已注册表达';
    status.textContent=({ACCEPT:'已识别',CONFIRM:'待确认',REJECT:'未匹配'})[item.status];row.append(number,time,text,status);list.append(row);
  });
}
function renderContinuousBuilder(){
  const list=$("queueList");list.replaceChildren();
  if(state.session!=="sequence"){set("queueSummary","等待长语音样本");return;}
  const item=state.presets.find(p=>p.id===$("presetSelect").value);
  if(!item){set("queueSummary","尚未载入长语音");return;}
  const row=document.createElement("li"),n=document.createElement("span"),name=document.createElement("b"),remove=document.createElement("button");
  n.textContent="01";name.textContent=item.display_name||item.id;remove.textContent="×";remove.setAttribute("aria-label","清除当前长语音");remove.disabled=state.running||state.loading;
  remove.addEventListener("click",()=>{if(state.running||state.loading)return;$("presetSelect").value="";state.buffer=null;resetTask();renderContinuousBuilder();syncControls();set("sampleDescription","请选择一段长语音样本");});
  row.append(n,name,remove);list.append(row);
  set("queueSummary",state.buffer?`1 段 · ${state.buffer.duration.toFixed(1)} 秒 · Continuous`:`1 段 · 正在读取实际时长`);
}
function validateContinuousResult(result){
  if(!result||!Array.isArray(result.segments))throw new Error('长语音接口响应缺少 segments。');
  let previousEnd=0;
  result.segments.forEach((item,i)=>{
    if(!Number.isFinite(item.start)||!Number.isFinite(item.end)||item.start<0||item.end<=item.start)throw new Error(`第 ${i+1} 段时间边界无效。`);
    if(state.buffer&&item.end>state.buffer.duration+.1)throw new Error(`第 ${i+1} 段超出长语音时长。`);
    if(item.start<previousEnd-.08)throw new Error(`第 ${i+1} 段与上一段发生异常重叠。`);
    if(!['ACCEPT','CONFIRM','REJECT'].includes(item.status))throw new Error(`第 ${i+1} 段返回未知状态。`);
    if(item.status==='ACCEPT'&&(typeof item.canonical_text!=='string'||!item.canonical_text.trim()))throw new Error(`第 ${i+1} 段 ACCEPT 缺少规范文本。`);
    previousEnd=item.end;
  });
  return result;
}
async function switchSession(session){
  if(state.running||state.loading)return;
  state.session=session;document.body.dataset.session=session;
  document.querySelectorAll('.session-tab').forEach(b=>{b.classList.toggle('active',b.dataset.session===session);b.setAttribute('aria-pressed',String(b.dataset.session===session));});
  show('sequenceBuilder',session==='sequence');show('addQueueBtn',session==='sequence');
  setMode('instant');const available=populatePresetSelect();
  if(available)await selectPreset();else{state.buffer=null;resetTask();set('sampleDescription',session==='sequence'?'后端尚未提供 continuous 类型的长语音样本。':'暂无可用样本。');}
  if(session==='sequence'){set('heroTitle','准备长语音识别');set('heroCaption','整段音频一次提交，结果按时间轴逐段呈现。');}
  renderContinuousBuilder();syncControls();
}
async function recognizeContinuous(){
  if(state.running||!state.buffer||!$('presetSelect').value)return;
  clearResult();clearEvents();const token=++state.run,controller=new AbortController();state.controller=controller;state.running=true;
  const player=$('audioPlayer'),item=state.presets.find(p=>p.id===$('presetSelect').value);player.pause();player.currentTime=0;
  [...$('stageTrack').children].forEach(e=>e.className='');
  document.body.dataset.state='running';$('resultCard').setAttribute('aria-busy','true');syncControls();
  set('heroTitle','长语音识别请求已发送');set('heroCaption',state.mode==='presentation'?'接口即时请求；长语音流程说明按讲解节奏展开。':'正在处理整段连续语音；结果将按时间轴返回。');
  set('analysisBadge',state.mode==='presentation'?'讲解进行中':'连续识别中');set('decisionBadge','处理中');set('taskStatus','处理中');set('taskBadge','长语音');set('streamState','记录中');
  setClosure(0,'done','已接入');setClosure(1,'active','连续识别');setProgress(0);if(state.mode!=='presentation')set('progressText','—');
  addEvent(`已提交长语音：${item?.display_name||item?.id||'当前样本'} / ${state.buffer.duration.toFixed(2)} 秒。`,state.mock?'模拟请求':'已发送');
  let outcome=null;
  const started=performance.now();
  // Start immediately: presentation duration is never reported as backend latency.
  const pending=continuousApi(item,controller.signal).then(validateContinuousResult)
    .then(result=>{outcome={result,requestMs:performance.now()-started};if(token===state.run){setClosure(1,'done','已返回');addEvent(state.mock?'模拟长语音接口已返回固定结果。':'长语音接口已返回结果。','已返回');}})
    .catch(error=>{outcome={error};});
  try{
    await presentUntilOutcome(token,started,()=>outcome,CONTINUOUS_EXPLANATIONS,'长语音流程讲解','长语音识别请求进行中');
    await pending;
    if(token!==state.run)return;
    const result=outcome.result;
    state.continuousResults=[...result.segments].sort((a,b)=>a.start-b.start);renderContinuousRows();show('decisionWaiting',false);show('decisionResult');show('sequenceResults');
    const allAccepted=state.continuousResults.length>0&&state.continuousResults.every(r=>r.status==='ACCEPT');
    const acceptCount=state.continuousResults.filter(r=>r.status==='ACCEPT').length,confirmCount=state.continuousResults.filter(r=>r.status==='CONFIRM').length,rejectCount=state.continuousResults.filter(r=>r.status==='REJECT').length;
    const merged=state.continuousResults.filter(r=>r.status==='ACCEPT').map(r=>r.canonical_text.trim()).join('。');
    document.body.dataset.state=allAccepted?'accepted':'confirm';$('resultCard').dataset.result=allAccepted?'ACCEPT':'CONFIRM';
    set('decisionBadge',allAccepted?'已识别':'需核对');set('resultBadge',state.mock?(allAccepted?'长语音结果 · 模拟':'长语音结果 · 含不确定段'):allAccepted?'长语音识别完成':'长语音识别完成 · 含不确定段');
    set('outputLabel',allAccepted?'规范中文序列':'分段结果');set('canonicalText',allAccepted?merged:`共 ${state.continuousResults.length} 段：${acceptCount} 接受 / ${confirmCount} 待确认 / ${rejectCount} 拒识`);
    const latency=Number.isFinite(result.latency_ms)?result.latency_ms:outcome.requestMs;set('latencyText',Number.isFinite(result.latency_ms)?`${state.mock?'模拟耗时':'后端报告耗时'} ${result.latency_ms.toFixed(1)} ms`:`接口往返 ${latency.toFixed(1)} ms · 后端未提供 latency_ms`);
    state.tts=allAccepted?merged:'';$('speakBtn').disabled=!allAccepted||!supportsTts();$('copyResultBtn').disabled=!allAccepted;
    set('ttsHint',allAccepted?(supportsTts()?'全部分段均已确认，可合并播放普通话。':'全部分段均已确认，可复制规范中文。'):'存在 CONFIRM / REJECT，已阻止整段自动播报。');
    $('checkText').className=allAccepted?'done':'';$('checkText').querySelector('b').textContent=allAccepted?'已生成':'需核对';
    setClosure(1,'done',`${state.continuousResults.length} 段`);setClosure(2,allAccepted?'done':'',allAccepted?'已生成':'需核对');
    set('heroTitle',allAccepted?'长语音识别完成':'长语音识别完成 · 需要核对');set('heroCaption',`时间轴已返回：${acceptCount} ACCEPT / ${confirmCount} CONFIRM / ${rejectCount} REJECT。`);
    if(state.mode==='presentation')[...$('stageTrack').children].forEach(e=>e.className='done');
    set('analysisBadge','已完成');set('taskStatus','已完成');set('taskBadge','已完成');set('streamState','已完成');set('currentStage',state.mode==='presentation'?`长语音流程讲解结束 · ${state.continuousResults.length} 个时间片段`:`连续识别完成 · ${state.continuousResults.length} 个时间片段`);setProgress(1);renderVisuals(1);
    state.continuousResults.forEach((seg,i)=>addEvent(`第 ${i+1} 段 ${seg.start.toFixed(2)}–${seg.end.toFixed(2)}s：${seg.status}${seg.status==='ACCEPT'?` / ${seg.canonical_text}`:''}`,'时间轴',seg.status==='ACCEPT'?'ok':'warn'));
    if(allAccepted&&$('autoTts').checked)speak();
  }catch(error){if(token!==state.run)return;showError(error.message);addEvent(error.message,'长语音异常','warn');}
  finally{if(token===state.run){state.running=false;$('resultCard').removeAttribute('aria-busy');syncControls();}}
}

function bind(){
  $("presetSelect").addEventListener("change",selectPreset);$("recognizeBtn").addEventListener("click",recognize);$("restartBtn").addEventListener("click",recognize);$("cancelBtn").addEventListener("click",cancelRun);$("retryBootBtn").addEventListener("click",boot);
  const player=$("audioPlayer");let playbackFrame;
  player.addEventListener("play",()=>{set("playOriginalBtn","暂停原始语音");const paint=()=>{if(player.paused)return;renderVisuals(player.currentTime/Math.max(.01,player.duration));playbackFrame=requestAnimationFrame(paint);};cancelAnimationFrame(playbackFrame);paint();});
  for(const event of ["pause","ended"])player.addEventListener(event,()=>{cancelAnimationFrame(playbackFrame);set("playOriginalBtn","试听原始语音");});
  $("playOriginalBtn").addEventListener("click",async()=>{try{if(player.paused)await player.play();else player.pause();}catch{set("sampleDescription","浏览器未能播放音频，请检查声音输出或重试。");}});
  document.querySelectorAll(".mode-btn").forEach(button=>button.addEventListener("click",()=>{if(state.running)return;setMode(button.dataset.mode);}));
  document.querySelectorAll('.session-tab').forEach(button=>button.addEventListener('click',()=>switchSession(button.dataset.session)));
  $('addQueueBtn').addEventListener('click',async()=>{if(state.running||state.loading||!$('presetSelect').value)return;await selectPreset();addEvent('长语音样本已载入，可开始识别。','已准备');});
  $('fillQueueBtn').addEventListener('click',async()=>{if(state.running||state.loading)return;const item=state.presets.find(p=>p.demo_kind==='continuous');if(!item)return;$('presetSelect').value=item.id;await selectPreset();});
  $('clearQueueBtn').addEventListener('click',()=>{if(state.running||state.loading)return;resetTask();renderContinuousBuilder();syncControls();});
  $("clearStreamBtn").addEventListener("click",clearEvents);$("exportStreamBtn").addEventListener("click",exportEvents);$("pauseStreamBtn").addEventListener("click",()=>{state.paused=!state.paused;set("pauseStreamBtn",state.paused?"继续滚动":"暂停滚动");if(!state.paused)$("eventStream").scrollTop=$("eventStream").scrollHeight;});
  $("speakBtn").addEventListener("click",speak);$("stopSpeakBtn").addEventListener("click",stopTts);
  $("copyResultBtn").addEventListener("click",async()=>{if(!state.tts)return;try{await navigator.clipboard.writeText(state.tts);set("copyResultBtn","已复制 ✓");}catch{set("copyResultBtn","复制失败，请手动选择文字");}});
  $("fullscreenBtn").addEventListener("click",async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();else throw new Error();}catch{addEvent("浏览器不支持全屏，请使用浏览器全屏快捷键。","提示","warn");}});
  document.addEventListener("fullscreenchange",()=>$("fullscreenBtn").setAttribute("aria-label",document.fullscreenElement?"退出全屏":"全屏展示"));
  $("guideBtn").addEventListener("click",()=>$("guideDialog").showModal());$("closeGuideBtn").addEventListener("click",()=>$("guideDialog").close());
  document.addEventListener("keydown",e=>{if(e.key==="Escape"&&!$("guideDialog").open)cancelRun();if(e.key==="Enter"&&!e.repeat&&!$("guideDialog").open&&!["BUTTON","INPUT","SELECT","TEXTAREA","A"].includes(e.target.tagName)){e.preventDefault();recognize();}});
  window.addEventListener("resize",()=>renderVisuals(state.frameProgress));
  document.addEventListener("visibilitychange",()=>{if(document.hidden){player.pause();stopTts();}});
  if(!supportsTts()){$("autoTts").checked=false;$("autoTts").disabled=true;}
  // Prime voice discovery without speaking or requesting microphone access.
  if(supportsTts())speechSynthesis.getVoices();
}
function tick(){const d=new Date();set("clockDate",d.toLocaleDateString("zh-CN",{month:"2-digit",day:"2-digit",weekday:"short"}));set("clockTime",d.toLocaleTimeString("zh-CN",{hour12:false}));}
bind();tick();setInterval(tick,1000);renderVisuals(0);boot();
