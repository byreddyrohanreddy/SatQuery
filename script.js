'use strict';

/* ============================================================================
   SatQuery AI — shared frontend logic
   ----------------------------------------------------------------------------
   script.js detects which page is active (via a class on <body>) and runs
   the matching init function:

     body.page-upload  -> initUploadPage()    (index.html)
     body.page-results -> initResultsPage()   (results.html)

   BASE_URL is the single constant to change if the backend moves.
   ============================================================================ */

const BASE_URL = 'http://localhost:8000';

/* sessionStorage key used to hand the API response + images from page 1 to
   page 2. sessionStorage (not localStorage) is deliberate: results are
   per-tab, are cleared when the tab closes, and never persist imagery
   across sessions. */
const STORAGE_KEY = 'satquery_result';

const REQUEST_TIMEOUT_MS = 60000;   // backend model calls can be slow
const HEALTH_TIMEOUT_MS = 5000;

/* Sample demo sets. The image files themselves live in /samples/ and are
   fetched at runtime, which means the demos work when the folder is served
   by any static file server (they cannot work from file:// due to browser
   security restrictions — a friendly note is shown in that case). */
const SAMPLE_DIR = 'samples/';

const SAMPLE_SETS = {
    single: {
        label: 'Single Image Demo',
        files: { image1: 'sample_single_optical.jpg' },
        query: 'What land cover types are visible in this image?'
    },
    fusion: {
        label: 'Optical + SAR Demo',
        files: { image1: 'sample_optical.jpg', image2: 'sample_sar.jpg' },
        query: 'Combine the optical and SAR imagery and describe what each sensor reveals about this scene.'
    },
    change: {
        label: 'Change Detection Demo',
        files: { image1: 'sample_before.jpg', image2: 'sample_after.jpg' },
        query: 'What changed between these two dates? Highlight the changed regions.'
    }
};

/* Example queries rotated through the textarea placeholder (index page). */
const EXAMPLE_QUERIES = [
    'Highlight the water body in this image…',
    'What land cover types are visible in this scene?',
    'What changed between these two acquisition dates?'
];

/* ── Small DOM / async helpers ─────────────────────────────────────────── */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

/** Create an element with a class and (safely, via textContent) some text. */
function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
}

/** Show the inline error banner with a friendly message. */
function showError(textEl, bannerEl, message) {
    textEl.textContent = message;
    bannerEl.hidden = false;
    bannerEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/** Generate a compact thumbnail data URL to stay well within sessionStorage's 5MB quota. */
function fileToThumbnailUrl(file, maxDim = 800) {
    return new Promise((resolve) => {
        if (/\.(tiff?)$/i.test(file.name) || file.type === 'image/tiff') {
            resolve('');
            return;
        }
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            URL.revokeObjectURL(url);
            let w = img.naturalWidth || img.width;
            let h = img.naturalHeight || img.height;
            if (Math.max(w, h) > maxDim) {
                const scale = maxDim / Math.max(w, h);
                w = Math.round(w * scale);
                h = Math.round(h * scale);
            }
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);
            resolve(canvas.toDataURL('image/jpeg', 0.8));
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            resolve('');
        };
        img.src = url;
    });
}

/** Read a File as a data URL (fallback for tiny non-TIFF files). */
function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Could not read the selected image file.'));
        reader.readAsDataURL(file);
    });
}

/** fetch() wrapper with a timeout; rejects with a friendly Error. */
async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, Object.assign({}, options, { signal: controller.signal }));
    } catch (err) {
        if (err && err.name === 'AbortError') {
            throw new Error('The request timed out. The backend may still be processing — try again.');
        }
        throw new Error('Could not reach the backend at ' + BASE_URL +
            '. Is it running? (' + (err && err.message ? err.message : 'network error') + ')');
    } finally {
        clearTimeout(timer);
    }
}

/* ── Page router ───────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
    if (document.body.classList.contains('page-upload')) initUploadPage();
    if (document.body.classList.contains('page-results')) initResultsPage();
});

/* ══════════════════════════════════════════════════════════════════════════
   PAGE 1 — Upload / Query (index.html)
   ══════════════════════════════════════════════════════════════════════════ */

function initUploadPage() {
    const textarea = $('#queryText');
    const runBtn = $('#runBtn');
    const banner = $('#errorBanner');
    const bannerText = $('#errorText');
    const loading = $('#loadingPanel');
    const steps = $$('[data-step]');
    const sampleNote = $('#sampleNote');

    /* Live state for the two upload slots. Object URLs are revoked when a
       file is replaced or removed so we never leak memory. */
    const state = {
        image1: null, image2: null,
        urls: { image1: null, image2: null }
    };

    /* ── Drop zone wiring (drag & drop + click/keyboard browse) ────────── */

    function setupDropzone(slot) {
        const zone = $('#drop' + slot.slice(-1));
        const input = $('#file' + slot.slice(-1));

        zone.addEventListener('click', () => input.click());
        zone.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
        });
        input.addEventListener('change', () => {
            if (input.files && input.files[0]) setFile(slot, input.files[0]);
            input.value = ''; // allow re-selecting the same file later
        });

        ['dragenter', 'dragover'].forEach((evt) =>
            zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add('drag-over'); }));
        ['dragleave', 'drop'].forEach((evt) =>
            zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove('drag-over'); }));
        zone.addEventListener('drop', (e) => {
            const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (file) setFile(slot, file);
        });

        $('#remove' + slot.slice(-1)).addEventListener('click', (e) => {
            e.stopPropagation(); // don't reopen the file picker
            clearFile(slot);
        });
    }

    function setFile(slot, file) {
        const okTypes = ['image/jpeg', 'image/png', 'image/tiff'];
        const okExt = /\.(jpe?g|png|tiff?)$/i;
        if (okTypes.indexOf(file.type) === -1 && !okExt.test(file.name)) {
            showError(bannerText, banner,
                '"' + file.name + '" does not look like an image. Please use JPG, PNG, or TIFF.');
            return;
        }
        if (state.urls[slot]) URL.revokeObjectURL(state.urls[slot]);
        state[slot] = file;
        state.urls[slot] = URL.createObjectURL(file);

        const thumb = $('#thumb' + slot.slice(-1));
        const isTiff = /\.(tiff?)$/i.test(file.name);
        const TIFF_SVG = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 200'%3E%3Crect width='300' height='200' fill='%231F497D' rx='8'/%3E%3Ctext x='50%25' y='42%25' dominant-baseline='middle' text-anchor='middle' fill='%23ffffff' font-family='sans-serif' font-size='22' font-weight='bold'%3ETIFF / GeoTIFF%3C/text%3E%3Ctext x='50%25' y='65%25' dominant-baseline='middle' text-anchor='middle' fill='%23EAF1FB' font-family='sans-serif' font-size='13'%3EReady for Analysis%3C/text%3E%3C/svg%3E";

        thumb.onerror = () => { thumb.src = TIFF_SVG; };
        thumb.src = isTiff ? TIFF_SVG : state.urls[slot];

        $('#name' + slot.slice(-1)).textContent = file.name;
        $('#body' + slot.slice(-1)).hidden = true;
        $('#preview' + slot.slice(-1)).hidden = false;
        banner.hidden = true;
        updateRunButton();
    }

    function clearFile(slot) {
        if (state.urls[slot]) URL.revokeObjectURL(state.urls[slot]);
        state[slot] = null;
        state.urls[slot] = null;
        $('#preview' + slot.slice(-1)).hidden = true;
        $('#body' + slot.slice(-1)).hidden = false;
        updateRunButton();
    }

    function updateRunButton() {
        runBtn.disabled = !(state.image1 && textarea.value.trim().length > 0);
    }

    setupDropzone('image1');
    setupDropzone('image2');
    textarea.addEventListener('input', updateRunButton);
    $('#errorClose').addEventListener('click', () => { banner.hidden = true; });

    /* ── Rotating placeholder examples ─────────────────────────────────── */

    let exampleIdx = 0;
    textarea.placeholder = EXAMPLE_QUERIES[0];
    setInterval(() => {
        if (textarea.value.trim() === '') {
            exampleIdx = (exampleIdx + 1) % EXAMPLE_QUERIES.length;
            textarea.placeholder = EXAMPLE_QUERIES[exampleIdx];
        }
    }, 3500);

    /* ── Optional backend health check (header status pill) ────────────── */

    (async () => {
        const dot = $('#statusDot'), txt = $('#statusText');
        try {
            const res = await fetchWithTimeout(BASE_URL + '/health', {}, HEALTH_TIMEOUT_MS);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            dot.className = 'status-dot status-ok';
            txt.textContent = 'API connected (' + BASE_URL + ')';
        } catch (err) {
            dot.className = 'status-dot status-bad';
            txt.textContent = 'API unreachable — start the backend, then reload';
        }
    })();

    /* ── Sample demo buttons ───────────────────────────────────────────── */

    $$('[data-sample]').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const set = SAMPLE_SETS[btn.dataset.sample];
            if (!set) return;
            btn.disabled = true;
            sampleNote.hidden = true;
            try {
                for (const slot of ['image1', 'image2']) {
                    const fname = set.files[slot];
                    if (!fname) { clearFile(slot); continue; }
                    const res = await fetch(SAMPLE_DIR + fname);
                    if (!res.ok) throw new Error('Sample file "' + fname + '" was not found in /samples/.');
                    const blob = await res.blob();
                    const file = new File([blob], fname, { type: blob.type || 'image/jpeg' });
                    setFile(slot, file);
                }
                textarea.value = set.query;
                updateRunButton();
            } catch (err) {
                sampleNote.textContent =
                    'Could not load the sample files (' + err.message + ') ' +
                    'If you opened this page directly from disk, serve the folder instead, ' +
                    'e.g. "python -m http.server" from the project root, then reload.';
                sampleNote.hidden = false;
            } finally {
                btn.disabled = false;
            }
        });
    });

    /* ── Agentic loading sequence ────────────────────────────────────────
       Three labeled steps are revealed in order with timed CSS transitions,
       reinforcing the "agent at work" narrative instead of a generic spinner.
       When the fetch resolves, all remaining steps snap to "done" and the
       page navigates — the user sees the full pipeline complete.           */

    function startStepSequence() {
        steps.forEach((s) => s.classList.remove('active', 'done'));
        loading.hidden = false;
        steps.forEach((step, i) => {
            setTimeout(() => {
                step.classList.add('active');
                for (let j = 0; j < i; j++) {
                    steps[j].classList.remove('active');
                    steps[j].classList.add('done');
                }
            }, 500 + i * 800);
        });
    }

    function finishStepSequence() {
        steps.forEach((s) => { s.classList.remove('active'); s.classList.add('done'); });
    }

    function resetAfterFailure() {
        loading.hidden = true;
        runBtn.disabled = false;
    }

    /* ── Submit → POST /query → sessionStorage handoff → navigate ───────── */

    runBtn.addEventListener('click', async () => {
        banner.hidden = true;
        runBtn.disabled = true;
        startStepSequence();

        try {
            const formData = new FormData();
            formData.append('image1', state.image1);
            if (state.image2) formData.append('image2', state.image2);
            formData.append('query', textarea.value.trim());

            const res = await fetchWithTimeout(BASE_URL + '/query',
                { method: 'POST', body: formData }, REQUEST_TIMEOUT_MS);

            let payload = null;
            try { payload = await res.json(); }
            catch (parseErr) { /* leave payload null; handled below */ }

            if (!res.ok) {
                // 4xx responses (e.g. "this task needs 2 images") are shown as
                // friendly clarification messages, not crashes.
                const detail = (payload && (payload.detail || payload.message || payload.error)) ||
                    'The backend returned HTTP ' + res.status + '.';
                throw new Error(detail);
            }
            if (!payload || typeof payload.answer === 'undefined') {
                throw new Error('The backend returned an unexpected response shape (no "answer" field).');
            }

            /* Prepare lightweight image descriptors for sessionStorage.
               We use backend-generated previews when available (ideal for GeoTIFF/TIFF),
               or a downsampled canvas thumbnail. This ensures we never exceed
               the browser's 5MB sessionStorage quota. */
            const images = [];
            let imgIdx = 0;
            for (const slot of ['image1', 'image2']) {
                if (state[slot]) {
                    let previewUrl = (payload.image_previews && payload.image_previews[imgIdx]) ? payload.image_previews[imgIdx] : '';
                    if (!previewUrl && state[slot].size < 2000000) {
                        try {
                            previewUrl = await fileToThumbnailUrl(state[slot], 800);
                        } catch (e) {
                            previewUrl = '';
                        }
                    }
                    images.push({
                        name: state[slot].name,
                        dataUrl: previewUrl
                    });
                    imgIdx++;
                }
            }

            const itemToStore = {
                response: payload,
                images: images,
                query: textarea.value.trim(),
                sentAt: new Date().toISOString()
            };

            try {
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(itemToStore));
            } catch (quotaErr) {
                console.warn('sessionStorage quota exceeded; trimming payload and retrying:', quotaErr);
                try {
                    // Strip dataUrl strings to guarantee it fits under quota
                    itemToStore.images = images.map(im => ({ name: im.name, dataUrl: '' }));
                    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(itemToStore));
                } catch (retryErr) {
                    console.error('Failed to store in sessionStorage:', retryErr);
                }
            }

            finishStepSequence();
            setTimeout(() => { window.location.href = 'results.html'; }, 500);

        } catch (err) {
            resetAfterFailure();
            showError(bannerText, banner, err.message || 'Something went wrong while running the query.');
        }
    });
}

/* ══════════════════════════════════════════════════════════════════════════
   PAGE 2 — Results (results.html)
   ══════════════════════════════════════════════════════════════════════════ */

function initResultsPage() {
    const main = $('#resultsMain');

    /* ── Read the handoff payload from sessionStorage ────────────────────
       Wrapped in try/catch: corrupt or oversized data must degrade to the
       friendly empty state, never to a broken page.                       */
    let stored = null;
    try {
        const raw = sessionStorage.getItem(STORAGE_KEY);
        if (raw) stored = JSON.parse(raw);
    } catch (err) { stored = null; }

    if (!stored || !stored.response) {
        renderEmptyState(main);
        return;
    }

    const resp = stored.response;
    const images = stored.images || [];
    const trace = resp.execution_trace || {};

    /* 1 ── Imagery + visual evidence ────────────────────────────────────── */

    const imageCard = el('section', 'card');
    imageCard.setAttribute('aria-labelledby', 'imageryHeading');
    imageCard.appendChild(el('h2', null, '')).id = 'imageryHeading';
    imageCard.querySelector('h2').textContent = 'Imagery & visual evidence';

    const row = el('div', 'image-row');
    imageCard.appendChild(row);

    // Label pair is inferred from the task the agent selected.
    const labels = labelsForTask(trace.task_selected, images.length);
    images.forEach((img, idx) => {
        const fig = el('figure', 'image-figure');
        fig.style.position = 'relative'; // anchor for the overlay

        const im = el('img');
        const fallbackSvg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 200'%3E%3Crect width='300' height='200' fill='%231F497D' rx='6'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23ffffff' font-family='sans-serif' font-size='18'%3ESatellite Imagery%3C/text%3E%3C/svg%3E";
        const displaySrc = (resp.image_previews && resp.image_previews[idx]) || img.dataUrl || fallbackSvg;
        im.src = displaySrc;
        im.alt = labels[idx] || ('Image ' + (idx + 1));
        fig.appendChild(im);

        // Which boxes go on which image:
        //   bounding_box    -> primary image only (index 0)
        //   change_regions  -> both images (the regions refer to the same scene)
        const ev = resp.visual_evidence || {};
        if (Array.isArray(ev.boxes) && ev.boxes.length > 0) {
            const onThisImage =
                ev.type === 'change_regions' ? true :
                    ev.type === 'bounding_box' ? idx === 0 : false;
            if (onThisImage) attachOverlay(fig, im, ev.boxes, ev.type);
        }

        const cap = el('figcaption', null, labels[idx] || ('Image ' + (idx + 1)));
        fig.appendChild(cap);
        row.appendChild(fig);
    });
    main.appendChild(imageCard);

    /* 2 ── Answer card ──────────────────────────────────────────────────── */

    const answerCard = el('section', 'card answer-card');
    answerCard.setAttribute('aria-labelledby', 'answerHeading');
    const answerH = el('h2', null, 'Answer');
    answerH.id = 'answerHeading';
    answerCard.appendChild(answerH);
    answerCard.appendChild(el('p', 'answer-text', resp.answer || '(no answer returned)'));
    main.appendChild(answerCard);

    /* 3 ── Confidence indicator (never show the number without its basis) ── */

    const confCard = el('section', 'card');
    confCard.setAttribute('aria-labelledby', 'confHeading');
    const confH = el('h2', null, 'Confidence');
    confH.id = 'confHeading';
    confCard.appendChild(confH);

    const confRow = el('div', 'conf-row');
    const conf = typeof resp.confidence === 'number' ? resp.confidence : null;
    const band = conf === null ? 'mid' : conf >= 0.7 ? 'high' : conf >= 0.4 ? 'mid' : 'low';

    if (conf !== null) {
        const pct = Math.round(conf * 100);
        confRow.appendChild(el('span', 'conf-badge conf-' + band, pct + '%'));

        const track = el('div', 'conf-track');
        track.setAttribute('role', 'img');
        track.setAttribute('aria-label', 'Confidence ' + pct + ' percent');
        const fill = el('div', 'conf-fill conf-' + band);
        track.appendChild(fill);
        confRow.appendChild(track);
        // Animate the bar after it is in the DOM.
        requestAnimationFrame(() => { fill.style.width = pct + '%'; });
    } else {
        confRow.appendChild(el('span', 'muted', 'No confidence value was returned.'));
    }

    if (resp.confidence_basis) {
        confRow.appendChild(el('p', 'conf-basis', 'Basis: ' + resp.confidence_basis));
    }
    confCard.appendChild(confRow);
    main.appendChild(confCard);

    /* 4 ── Execution trace panel — visible by default, not collapsed ────── */

    const traceCard = el('section', 'card trace-card');
    traceCard.setAttribute('aria-labelledby', 'traceHeading');
    const traceH = el('h2', null, 'How this answer was produced');
    traceH.id = 'traceHeading';
    traceCard.appendChild(traceH);

    traceCard.appendChild(traceRow('Task selected', humanizeTask(trace.task_selected)));

    // Routing path as a colored pill; fallback reason shown alongside it.
    const routeVal = el('div', 'trace-val');
    if (trace.routing_path) {
        const isFallback = trace.routing_path === 'fallback_routed';
        routeVal.appendChild(el('span', 'route-pill ' + (isFallback ? 'route-fallback' : 'route-llm'),
            isFallback ? 'Fallback routed' : 'LLM routed'));
        if (isFallback && trace.fallback_reason) {
            routeVal.appendChild(document.createTextNode(' '));
            routeVal.appendChild(el('span', 'muted small', '(' + trace.fallback_reason.replace(/_/g, ' ') + ')'));
        }
    } else {
        routeVal.textContent = 'Not reported';
    }
    traceCard.appendChild(traceRowEl('Routing path', routeVal));

    traceCard.appendChild(traceRow('Model(s) called', trace.model_called || 'Not reported'));

    // Validator output: image count, modality chips, warnings.
    const valVal = el('div', 'trace-val');
    const v = trace.validator_output;
    if (v) {
        valVal.appendChild(el('span', null,
            (v.image_count !== undefined ? v.image_count : '?') + ' image(s) detected'));
        valVal.appendChild(document.createTextNode(' '));
        (v.modalities || []).forEach((m) => valVal.appendChild(el('span', 'chip', m)));
        if (Array.isArray(v.warnings) && v.warnings.length > 0) {
            const ul = el('ul', 'warn-list');
            v.warnings.forEach((w) => ul.appendChild(el('li', null, w)));
            valVal.appendChild(ul);
        }
    } else {
        valVal.textContent = 'Not reported';
    }
    traceCard.appendChild(traceRowEl('Validator output', valVal));

    // Parameters used, rendered as a compact key: value list.
    const paramVal = el('div', 'trace-val');
    const params = trace.parameters_used;
    if (params && Object.keys(params).length > 0) {
        const ul = el('ul', 'param-list');
        Object.keys(params).forEach((k) => {
            ul.appendChild(el('li', null, k + ': ' + JSON.stringify(params[k])));
        });
        paramVal.appendChild(ul);
    } else {
        paramVal.textContent = 'None reported';
    }
    traceCard.appendChild(traceRowEl('Parameters used', paramVal));

    let tsText = 'Not reported';
    if (trace.timestamp) {
        try { tsText = new Date(trace.timestamp).toLocaleString(); }
        catch (e) { tsText = trace.timestamp; }
    }
    traceCard.appendChild(traceRow('Timestamp', tsText));

    main.appendChild(traceCard);

    /* 5 ── Run another query (also clears the stored result) ────────────── */

    const againRow = el('div', 'again-row');
    const again = el('a', 'btn btn-primary', 'Run Another Query');
    again.href = '/';
    again.target = '_self';
    again.addEventListener('click', () => sessionStorage.removeItem(STORAGE_KEY));
    againRow.appendChild(again);
    main.appendChild(againRow);
}

/* ── Results-page helpers ──────────────────────────────────────────────── */

/** Friendly empty state when someone lands on results.html directly. */
function renderEmptyState(main) {
    const card = el('section', 'card empty-state');
    card.appendChild(el('h2', null, 'No results to show yet'));
    card.appendChild(el('p', 'muted',
        'This page displays the output of a query run from the upload page. ' +
        'Run a query first, and the answer, evidence, confidence, and execution trace will appear here.'));
    const btn = el('a', 'btn btn-primary', 'Go to the upload page');
    btn.href = '/';
    btn.target = '_self';
    card.appendChild(btn);
    main.appendChild(card);
}

/** Map the agent's task selection to the right label pair for the images. */
function labelsForTask(task, imageCount) {
    if (imageCount < 2) return ['Image 1'];
    if (task === 'change_analysis') return ['Before', 'After'];
    if (task === 'optical_sar_fusion') return ['Optical', 'SAR'];
    return ['Image 1', 'Image 2'];
}

/** 'vqa_caption' -> 'VQA / Captioning', etc. */
function humanizeTask(task) {
    const map = {
        vqa_caption: 'VQA / Captioning',
        grounding: 'Grounding (object localization)',
        change_analysis: 'Change analysis',
        optical_sar_fusion: 'Optical + SAR fusion'
    };
    return map[task] || (task || 'Not reported');
}

/** Build a labeled trace row: <div class="trace-row"> with key + value node. */
function traceRowEl(key, valueNode) {
    const row = el('div', 'trace-row');
    row.appendChild(el('div', 'trace-key', key));
    if (typeof valueNode === 'string') row.appendChild(el('div', 'trace-val', valueNode));
    else row.appendChild(valueNode);
    return row;
}
function traceRow(key, valueText) { return traceRowEl(key, String(valueText)); }

/**
 * Visual-evidence overlay — the trickiest part of this page.
 *
 * API boxes use NORMALIZED coordinates (x, y, width, height each in 0-1,
 * fractions of the image's true pixel dimensions). To draw them over the
 * image as displayed in the browser we:
 *
 *   1. Create a zero-children <div class="evidence-overlay"> inside the
 *      <figure> (which is position:relative) and size it — in DISPLAYED
 *      pixels — to exactly match the <img>'s rendered box via
 *      img.clientWidth / img.clientHeight.
 *   2. Position each box with CSS percentages of that overlay
 *      (left: x*100%, top: y*100%, width: w*100%, height: h*100%). Because
 *      an image scales uniformly with its container, the percentage box
 *      tracks the same underlying pixels no matter how the window resizes —
 *      but we still re-sync the overlay's pixel size on 'resize' and on
 *      image 'load' so it can never drift from the rendered image.
 *
 * The overlay has pointer-events:none so it never blocks interaction with
 * the image underneath.
 */
function attachOverlay(figure, img, boxes, evidenceType) {
    const overlay = document.createElement('div');
    overlay.className = 'evidence-overlay';
    figure.appendChild(overlay);

    function draw() {
        overlay.style.width = img.clientWidth + 'px';
        overlay.style.height = img.clientHeight + 'px';
        overlay.innerHTML = '';

        boxes.forEach((b) => {
            const box = el('div', 'ev-box' + (evidenceType === 'change_regions' ? ' ev-box-change' : ''));
            box.style.left = (b.x * 100) + '%';
            box.style.top = (b.y * 100) + '%';
            box.style.width = (b.width * 100) + '%';
            box.style.height = (b.height * 100) + '%';

            if (b.label) {
                const tag = el('span', 'ev-label', b.label);
                box.appendChild(tag);
            }
            overlay.appendChild(box);
        });
    }

    if (img.complete && img.naturalWidth > 0) draw();
    else img.addEventListener('load', draw);

    // Re-sync overlay size if the layout changes (window resize, rotation).
    let resizeTimer = null;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(draw, 100); // debounced; keeps math cheap
    });
}