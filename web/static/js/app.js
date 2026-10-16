/**
 * AstroClassify Studio - Frontend Interactive Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("classifier-form");
    const presetsContainer = document.getElementById("presets-container");
    const btnReset = document.getElementById("btn-reset");

    // Elements
    const topTitle = document.getElementById("top-title");
    const topDesc = document.getElementById("top-desc");
    const topIcon = document.getElementById("top-icon");
    const confBar = document.getElementById("conf-bar");
    const confVal = document.getElementById("conf-val");
    const probList = document.getElementById("prob-list");
    const predBadge = document.getElementById("pred-badge");

    const valUg = document.getElementById("val-ug");
    const valGr = document.getElementById("val-gr");
    const valRi = document.getElementById("val-ri");
    const valPm = document.getElementById("val-pm");

    const sedLine = document.getElementById("sed-line");
    const pU = document.getElementById("p-u");
    const pG = document.getElementById("p-g");
    const pR = document.getElementById("p-r");
    const pI = document.getElementById("p-i");
    const pZ = document.getElementById("p-z");

    const CLASS_METADATA = {
        "quasar": { icon: "🌀", title: "Quasar (QSO)", desc: "Extremely luminous active galactic nucleus at high redshift with strong broad emission lines." },
        "white_dwarf": { icon: "⚪", title: "White Dwarf", desc: "Dense stellar core remnant with high surface gravity, high proper motion, and blue excess." },
        "red_giant": { icon: "🔴", title: "Red Giant", desc: "Evolved luminous giant star with high infrared flux and low effective surface temperature." },
        "main_sequence_star": { icon: "⭐", title: "Main Sequence Star", desc: "Hydrogen-burning dwarf star with balanced photometric colors and typical kinematics." },
        "galaxy": { icon: "🌌", title: "Galaxy", desc: "Extended composite extragalactic system with negligible stellar proper motion and cosmological redshift." },
        "exoplanet_candidate": { icon: "🪐", title: "Exoplanet Candidate Host", desc: "Photometrically stable bright stellar target displaying periodic transit signatures." }
    };

    let presetsData = {};

    // Load Presets
    async function initPresets() {
        try {
            const res = await fetch("/api/presets");
            presetsData = await res.json();
            renderPresetButtons();
        } catch (e) {
            console.error("Failed to fetch presets:", e);
        }
    }

    function renderPresetButtons() {
        presetsContainer.innerHTML = "";
        Object.entries(presetsData).forEach(([key, preset]) => {
            const btn = document.createElement("button");
            btn.className = "btn-preset";
            btn.type = "button";
            btn.innerHTML = `<span>${CLASS_METADATA[key]?.icon || '✨'}</span> ${preset.title}`;
            btn.addEventListener("click", () => applyPreset(key, btn));
            presetsContainer.appendChild(btn);
        });
    }

    function applyPreset(key, btnElem) {
        document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
        if (btnElem) btnElem.classList.add("active");

        const data = presetsData[key];
        if (!data) return;

        Object.keys(data).forEach(field => {
            const input = document.getElementById(field);
            if (input) input.value = data[field];
        });

        triggerClassification();
    }

    function getFormData() {
        const payload = {};
        new FormData(form).forEach((val, key) => {
            payload[key] = parseFloat(val) || 0;
        });
        return payload;
    }

    async function triggerClassification() {
        const payload = getFormData();
        updateSedCurve(payload);

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            renderResults(data);
        } catch (e) {
            console.error("Classification error:", e);
        }
    }

    function renderResults(data) {
        const topKey = data.top_class;
        const meta = CLASS_METADATA[topKey] || { icon: "✨", title: topKey, desc: "Astronomical object" };

        topTitle.textContent = meta.title;
        topDesc.textContent = meta.desc;
        topIcon.textContent = meta.icon;

        const conf = data.confidence;
        confVal.textContent = `${conf}%`;
        confBar.style.width = `${Math.min(100, Math.max(5, conf))}%`;
        predBadge.textContent = `${meta.title} (${conf}%)`;

        // Probabilities
        probList.innerHTML = "";
        Object.entries(data.probabilities).forEach(([cls, prob]) => {
            const item = document.createElement("div");
            item.className = "prob-item";
            const cMeta = CLASS_METADATA[cls] || { icon: "", title: cls };
            item.innerHTML = `
                <div class="prob-row">
                    <span class="prob-name">${cMeta.icon} ${cMeta.title}</span>
                    <span class="prob-val">${prob}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${prob}%;"></div>
                </div>
            `;
            probList.appendChild(item);
        });

        // Derived features
        const df = data.derived_features || {};
        valUg.textContent = df.u_g_color?.toFixed(3) || "0.000";
        valGr.textContent = df.g_r_color?.toFixed(3) || "0.000";
        valRi.textContent = df.r_i_color?.toFixed(3) || "0.000";
        valPm.textContent = `${df.total_proper_motion?.toFixed(3) || "0.000"} mas/yr`;
    }

    function updateSedCurve(p) {
        // Map magnitudes (e.g. 12 to 20) to SVG y-coordinates (higher mag = fainter = higher y)
        const mags = [p.u_mag || 18, p.g_mag || 18, p.r_mag || 17.5, p.i_mag || 17, p.z_mag || 16.5];
        const minM = 11.0;
        const maxM = 21.0;

        const getY = (m) => {
            const norm = (m - minM) / (maxM - minM);
            const clamped = Math.min(1.0, Math.max(0.0, norm));
            return 20 + clamped * 90; // y from 20 (brightest) to 110 (faintest)
        };

        const pts = [
            { x: 40, y: getY(mags[0]) },
            { x: 120, y: getY(mags[1]) },
            { x: 200, y: getY(mags[2]) },
            { x: 280, y: getY(mags[3]) },
            { x: 360, y: getY(mags[4]) }
        ];

        sedLine.setAttribute("points", pts.map(pt => `${pt.x},${pt.y.toFixed(1)}`).join(" "));
        pU.setAttribute("cy", pts[0].y);
        pG.setAttribute("cy", pts[1].y);
        pR.setAttribute("cy", pts[2].y);
        pI.setAttribute("cy", pts[3].y);
        pZ.setAttribute("cy", pts[4].y);
    }

    // Event listeners
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        triggerClassification();
    });

    form.querySelectorAll("input").forEach(input => {
        input.addEventListener("input", () => {
            document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
            triggerClassification();
        });
    });

    btnReset.addEventListener("click", () => {
        if (presetsData["main_sequence_star"]) {
            applyPreset("main_sequence_star");
        }
    });

    // Start
    initPresets().then(() => {
        triggerClassification();
    });
});
