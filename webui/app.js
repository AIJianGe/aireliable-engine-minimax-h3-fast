/* MiniMaxH3Fast 引擎 WebUI（交互式控制台）。
 *
 * ⚠️ 关键：本页由引擎在 /ui/ 提供（页面 URL 是 `/ui/`），
 *    同源 fetch 必须用**根绝对路径**（`/capabilities` 而非 `capabilities`），
 *    否则相对路径会被解析成 `/ui/capabilities` → 404。
 *
 * 交互：状态轮询 / 加载/卸载模型 / 生成视频（POST /infer SSE 流式）。
 */
(function () {
    'use strict';

    function $(id) { return document.getElementById(id); }

    var dot = $('state-dot');
    var meta = $('engine-meta');
    var stateText = $('state-text');
    var statusLine = $('status-line');
    var logEl = $('log');
    var progBar = $('prog-bar');
    var video = $('video');
    var btnGen = $('btn-gen');

    function setState(kind, text) {
        dot.className = 'dot' + (kind ? ' ' + kind : '');
        stateText.textContent = text || '';
    }

    function log(msg, cls) {
        var line = document.createElement('div');
        line.className = cls || 'info';
        line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
        logEl.appendChild(line);
        logEl.scrollTop = logEl.scrollHeight;
    }
    function clearLog() { logEl.innerHTML = ''; }
    function setProg(p) { progBar.style.width = (p || 0) + '%'; }

    /* ---------- 状态 ---------- */
    function loadStatus() {
        fetch('/status').then(function (r) { return r.json(); })
            .then(function (d) {
                var s = String(d.state || '').toLowerCase();
                if (s === 'ready') setState('ready', '就绪');
                else if (s === 'error') setState('error', '异常');
                else if (s === 'inferring' || s === 'loading') setState('busy', s);
                else setState('', '空闲');
                var extra = d.extra || {};
                statusLine.innerHTML = 'state: <code>' + s + '</code>'
                    + (d.loaded_loras && d.loaded_loras.length ? ' · LoRA: ' + d.loaded_loras.join(',') : '')
                    + (extra.backend ? ' · backend: ' + extra.backend : '')
                    + (extra.lora_injected ? ' · 蒸馏LoRA已注入' : '');
            })
            .catch(function (e) { setState('error', '无法连接'); statusLine.textContent = '状态读取失败：' + e.message; });
    }

    /* ---------- 模型加载/卸载 ---------- */
    function doPost(url, label) {
        setState('busy', '处理中…');
        log('发起 ' + label + '…', 'info');
        return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
            .then(function (r) { return r.json().catch(function () { return { detail: r.statusText }; }).then(function (b) { return { status: r.status, body: b }; }); })
            .then(function (res) {
                if (res.status >= 400) {
                    var d = res.body && res.body.detail ? res.body.detail : 'HTTP ' + res.status;
                    log(label + ' 失败：' + d, 'err');
                    setState('error', '失败');
                    return { ok: false, detail: d };
                }
                log(label + ' 完成', 'ok');
                loadStatus();
                return { ok: true };
            })
            .catch(function (e) { log(label + ' 异常：' + e.message, 'err'); setState('error', '异常'); return { ok: false }; });
    }

    /* ---------- 生成视频（SSE 流式 POST /infer） ---------- */
    function generate() {
        var prompt = $('prompt').value.trim();
        if (!prompt) { log('请先填写 prompt。', 'err'); return; }
        var wh = $('res').value.split('x').map(Number);
        var payload = {
            variant: null,
            inputs: {
                prompt: prompt,
                negative_prompt: $('nprompt').value.trim(),
                width: wh[0],
                height: wh[1],
                num_frames: parseInt($('frames').value, 10) || 121,
                num_inference_steps: parseInt($('steps').value, 10) || 8,
                seed: parseInt($('seed').value, 10) || -1
            }
        };

        btnGen.disabled = true;
        video.style.display = 'none';
        video.removeAttribute('src');
        clearLog();
        setProg(2);
        setState('busy', '生成中…');
        log('开始生成：' + payload.inputs.prompt.slice(0, 40) + (prompt.length > 40 ? '…' : ''), 'info');

        fetch('/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }).then(function (resp) {
            if (!resp.ok) { return resp.json().then(function (b) { throw new Error((b && b.detail) ? b.detail : 'HTTP ' + resp.status); }); }
            var reader = resp.body.getReader();
            var decoder = new TextDecoder();
            var buf = '';
            function pump() {
                return reader.read().then(function (r) {
                    if (r.done) { setProg(100); setState('', '空闲'); loadStatus(); btnGen.disabled = false; return; }
                    buf += decoder.decode(r.value, { stream: true });
                    var parts = buf.split('\n\n');
                    buf = parts.pop() || '';
                    parts.forEach(function (chunk) { handleSse(chunk); });
                    return pump();
                });
            }
            function handleSse(chunk) {
                chunk.split('\n').forEach(function (line) {
                    if (line.indexOf('data:') !== 0) { return; }
                    var raw = line.slice(5).trim();
                    if (!raw) { return; }
                    var ev;
                    try { ev = JSON.parse(raw); } catch (e) { log('SSE 解析失败：' + raw.slice(0, 80), 'err'); return; }
                    var type = ev.type || ev.event || '';
                    if (ev.progress !== undefined) { setProg(ev.progress); }
                    if (ev.message) { log('  ' + ev.message, 'info'); }
                    else if (type === 'progress' || type === 'transform') { log('  进度…', 'info'); }
                    var result = ev.result || ev.data;
                    if (result && result.videos && result.videos.length) {
                        var v = result.videos[0];
                        var u = v.video_url || v.video_path;
                        log('完成：' + v.width + '×' + v.height + ' @' + (v.duration || '?') + 's', 'ok');
                        if (u) { video.src = u; video.style.display = 'block'; video.play().catch(function () {}); }
                    }
                    if (ev.status === 'failed' || ev.level === 'error') { log('任务失败', 'err'); }
                });
            }
            return pump();
        }).catch(function (e) {
            log('生成异常：' + e.message, 'err');
            setState('error', '异常');
            btnGen.disabled = false;
        });
    }

    /* ---------- init ---------- */
    function init() {
        fetch('/capabilities')
            .then(function (r) { return r.json(); })
            .then(function (d) {
                var caps = d.capabilities || [];
                var variants = d.variants || [];
                meta.textContent = d.engine_name + ' · ' + caps.length + ' 能力 · ' + variants.length + ' 档位';
            })
            .catch(function () { meta.textContent = '能力读取失败'; });
        $('btn-status').addEventListener('click', loadStatus);
        $('btn-load').addEventListener('click', function () { doPost('/load', '加载模型'); });
        $('btn-unload').addEventListener('click', function () {
            if (confirm('确定卸载模型？')) doPost('/unload', '卸载模型');
        });
        $('btn-gen').addEventListener('click', generate);
        loadStatus();
    }

    document.addEventListener('DOMContentLoaded', init);
})();