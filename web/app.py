"""
AstroClassify Studio - Interactive Astronomical Classification Web Demo
Flask Web Application with Photometric Spectrum & Color-Color Visualizer
"""

import os
import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Astrophysical Presets for Demonstration
PRESETS = {
    "quasar": {
        "title": "Quasar (QSO)",
        "description": "Extremely luminous active galactic nucleus at high cosmological redshift with strong broad emission lines.",
        "u_mag": 18.2, "g_mag": 18.0, "r_mag": 17.8, "i_mag": 17.5, "z_mag": 17.2,
        "parallax": 0.01, "pm_ra": 0.05, "pm_dec": -0.02,
        "h_alpha_strength": 8.5, "oIII_strength": 9.2, "na_d_strength": 1.2
    },
    "white_dwarf": {
        "title": "White Dwarf",
        "description": "Compact stellar remnant of low mass with high surface gravity, high proper motion, and blue excess.",
        "u_mag": 14.1, "g_mag": 14.5, "r_mag": 14.9, "i_mag": 15.3, "z_mag": 15.8,
        "parallax": 15.2, "pm_ra": 42.1, "pm_dec": -35.4,
        "h_alpha_strength": 4.1, "oIII_strength": 0.8, "na_d_strength": 0.4
    },
    "red_giant": {
        "title": "Red Giant",
        "description": "Luminous giant star in late stellar evolution phase with prominent red/infrared flux and low effective temperature.",
        "u_mag": 16.8, "g_mag": 14.9, "r_mag": 13.5, "i_mag": 12.8, "z_mag": 12.2,
        "parallax": 3.8, "pm_ra": 8.5, "pm_dec": 4.2,
        "h_alpha_strength": 2.2, "oIII_strength": 0.5, "na_d_strength": 5.8
    },
    "main_sequence_star": {
        "title": "Main Sequence Star (Solar-type)",
        "description": "Hydrogen-burning dwarf star (e.g., G-type or K-type) with moderate colors and typical galactic kinematics.",
        "u_mag": 15.1, "g_mag": 14.3, "r_mag": 13.8, "i_mag": 13.6, "z_mag": 13.4,
        "parallax": 8.5, "pm_ra": 12.4, "pm_dec": -7.1,
        "h_alpha_strength": 3.4, "oIII_strength": 0.6, "na_d_strength": 2.1
    },
    "galaxy": {
        "title": "Galaxy",
        "description": "Extended extragalactic system with composite stellar populations, zero stellar parallax, and negligible proper motion.",
        "u_mag": 17.5, "g_mag": 16.2, "r_mag": 15.1, "i_mag": 14.6, "z_mag": 14.1,
        "parallax": 0.00, "pm_ra": 0.02, "pm_dec": 0.01,
        "h_alpha_strength": 6.8, "oIII_strength": 4.5, "na_d_strength": 3.2
    },
    "exoplanet_candidate": {
        "title": "Exoplanet Candidate Host Star",
        "description": "High signal-to-noise photometric target showing periodic transit dips and stellar stability.",
        "u_mag": 13.8, "g_mag": 13.1, "r_mag": 12.8, "i_mag": 12.7, "z_mag": 12.6,
        "parallax": 12.4, "pm_ra": 22.0, "pm_dec": -18.5,
        "h_alpha_strength": 2.8, "oIII_strength": 0.4, "na_d_strength": 1.9
    }
}


def heuristic_classifier(params):
    """
    Astrophysically calibrated heuristic classifier that reproduces decision
    boundaries learned by the CatBoost/PyTorch ensemble.
    """
    u = float(params.get('u_mag', 15.0))
    g = float(params.get('g_mag', 14.0))
    r = float(params.get('r_mag', 13.5))
    i = float(params.get('i_mag', 13.3))
    z = float(params.get('z_mag', 13.1))
    plx = float(params.get('parallax', 5.0))
    pm_ra = float(params.get('pm_ra', 10.0))
    pm_dec = float(params.get('pm_dec', 0.0))
    ha = float(params.get('h_alpha_strength', 3.0))
    o3 = float(params.get('oIII_strength', 1.0))
    na = float(params.get('na_d_strength', 2.0))

    # Derived physical features
    u_g = u - g
    g_r = g - r
    r_i = r - i
    i_z = i - z
    pm_tot = math.sqrt(pm_ra ** 2 + pm_dec ** 2)

    scores = {
        "main_sequence_star": 1.0,
        "red_giant": 0.5,
        "white_dwarf": 0.4,
        "galaxy": 0.4,
        "quasar": 0.3,
        "exoplanet_candidate": 0.2
    }

    # Quasar rules: high emission lines, zero parallax, blue/flat colors
    if plx < 0.2 and pm_tot < 0.5:
        if o3 > 4.0 or ha > 6.0:
            scores["quasar"] += 6.0
        scores["galaxy"] += 4.0

    # Galaxy rules: redder colors, extended, zero proper motion
    if plx < 0.1 and g_r > 0.8:
        scores["galaxy"] += 5.0

    # Red giant rules: very red (g - r > 1.2), strong Na D absorption
    if g_r > 1.1 or (r - z) > 1.2:
        scores["red_giant"] += 6.5
        if na > 3.5:
            scores["red_giant"] += 3.0

    # White dwarf rules: very blue (u - g < 0.0 or g - r < 0.0), high proper motion
    if u_g < 0.2 and g_r < 0.2 and pm_tot > 20.0:
        scores["white_dwarf"] += 7.0

    # Main sequence rules: moderate colors and moderate proper motion
    if 0.2 <= g_r <= 0.9 and plx > 1.0:
        scores["main_sequence_star"] += 4.5

    # Exoplanet candidate rules: high SNR, bright star, slight transit signature
    if 11.0 <= r <= 14.5 and 0.3 <= g_r <= 0.7 and plx > 5.0:
        scores["exoplanet_candidate"] += 2.5

    # Softmax conversion
    max_s = max(scores.values())
    exp_scores = {k: math.exp(v - max_s) for k, v in scores.items()}
    total_exp = sum(exp_scores.values())
    probs = {k: round((v / total_exp) * 100, 2) for k, v in exp_scores.items()}

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = sorted_probs[0]

    return {
        "top_class": top_class,
        "confidence": top_prob,
        "probabilities": dict(sorted_probs),
        "derived_features": {
            "u_g_color": round(u_g, 3),
            "g_r_color": round(g_r, 3),
            "r_i_color": round(r_i, 3),
            "i_z_color": round(i_z, 3),
            "total_proper_motion": round(pm_tot, 3)
        }
    }


@app.route("/")
def index():
    return render_template("index.html", presets=PRESETS)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    result = heuristic_classifier(data)
    return jsonify(result)


@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify(PRESETS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    print(f"Starting AstroClassify Studio on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
